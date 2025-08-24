package convert

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/qrv0/crow/internal/quant"
)

type LayerSpec struct {
	Name   string
	Rows   int
	Cols   int
	Data   []float32 // row-major (rows*cols)
	Scope  uint16
}

type Config struct {
	Rank            int
	OutlierQuantile float64
	PQm             int
	PQk             int
}

type Shard struct {
	Type  uint8
	Scope uint16
	Comp  uint8
	Data  []byte
}

type Result struct {
	Shards []Shard
	Meta   map[string]any
}

// Decompose NDSQ: returns D, L, R, S
// Implementation: L via truncated SVD using gonum; D from diagonal of (W-L);
// S via outlier quantile from residual after removing L and D; R is remaining dense residual.
func decomposeNDSQ(rows, cols int, data []float32, rank int, outlierQ float64) (D, L, R []float32, Sind [][2]int32, Sval []float32, err error) {
	// Build float64 matrix for SVD
	A := make([]float64, rows*cols)
	for i := range data { A[i] = float64(data[i]) }
	// Compute thin SVD
	m := rows
	n := cols
	matA := mat.NewDense(m, n, A)
	var svd mat.SVD
	ok := svd.Factorize(matA, mat.SVDThin)
	if !ok {
		return nil, nil, nil, nil, nil, fmt.Errorf("svd factorization failed")
	}
	s := svd.Values(nil)
	r := rank
	if r > len(s) { r = len(s) }
	if r < 0 { r = 0 }
	// Get U_r and V_r
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)
	// Determine effective rank and slice U and V to first r columns
	Ur := U.Slice(0, m, 0, r)
	Vr := V.Slice(0, n, 0, r)
	// Construct L = U_r * S_r * V_r^T using diagonal S
	Sr := mat.NewDiagDense(r, s[:r])
	var tmp mat.Dense
	tmp.Mul(Ur, Sr)
	var L64 mat.Dense
	L64.Mul(&tmp, Vr.T())
	L = make([]float32, rows*cols)
	// copy float64 to float32
	for i, v := range L64.RawMatrix().Data {
		L[i] = float32(v)
	}
	// resid = W - L
	resid := make([]float32, rows*cols)
	for i := 0; i < rows*cols; i++ { resid[i] = data[i] - L[i] }
	// D: diagonal from resid
	D = make([]float32, rows*cols)
	minDim := rows
	if cols < rows { minDim = cols }
	for i := 0; i < minDim; i++ {
		idx := i*cols + i
		D[idx] = resid[idx]
		resid[idx] = 0
	}
	// S: outliers from resid via quantile
	abs := make([]float64, len(resid))
	for i := range resid { abs[i] = math.Abs(float64(resid[i])) }
	sorted := append([]float64(nil), abs...)
	sort.Float64s(sorted)
	th := 0.0
	if len(sorted) > 0 {
		qidx := int(float64(len(sorted)-1) * outlierQ)
		if qidx < 0 { qidx = 0 }
		if qidx >= len(sorted) { qidx = len(sorted)-1 }
		th = sorted[qidx]
	}
	mask := make([]bool, len(resid))
	cnt := 0
	for i, v := range abs {
		if v >= th && v != 0 {
			mask[i] = true
			cnt++
		}
	}
	Sind = make([][2]int32, 0, cnt)
	Sval = make([]float32, 0, cnt)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			if mask[idx] {
				Sind = append(Sind, [2]int32{int32(i), int32(j)})
				Sval = append(Sval, resid[idx])
				resid[idx] = 0
			}
		}
	}
	R = resid
	return
}

func fp16bytes(f []float32) []byte {
	buf := new(bytes.Buffer)
	for _, v := range f {
		buf.Write(fp32to16(v))
	}
	return buf.Bytes()
}

func fp32to16(f float32) []byte {
	// simple float32->float16 conversion (no denormal handling), little-endian
	u := math.Float32bits(f)
	sign := (u >> 31) & 0x1
	exp := (u >> 23) & 0xFF
	mant := u & 0x7FFFFF
	var h uint16
	if exp == 0 { // zero or subnormal -> zero
		h = uint16(sign << 15)
	} else if exp == 0xFF { // inf/nan -> inf
		h = uint16((sign << 15) | (0x1F << 10))
	} else {
		e := int(exp) - 127 + 15
		if e <= 0 { // underflow -> zero
			h = uint16(sign << 15)
		} else if e >= 0x1F { // overflow -> inf
			h = uint16((sign << 15) | (0x1F << 10))
		} else {
			h = uint16((sign << 15) | (uint32(e)&0x1F)<<10 | ((mant >> 13) & 0x3FF))
		}
	}
	b := []byte{byte(h & 0xFF), byte(h >> 8)}
	return b
}

// Pack a shard header and payload (type:uint8, scope:uint16, comp:uint8, usize:uint32, csize:uint32)
func packShard(t uint8, scope uint16, payload []byte) []byte {
	var hdr [12]byte
	// comp=0 raw
	hdr[0] = t
	hdr[1] = byte(scope & 0xFF)
	hdr[2] = byte(scope >> 8)
	hdr[3] = 0
	binary.LittleEndian.PutUint32(hdr[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(len(payload)))
	return append(hdr[:], payload...)
}

// Convert a single layer tensor
func ConvertLayer(spec LayerSpec, cfg Config) ([]Shard, error) {
	D, L, R, Sind, Sval, err := decomposeNDSQ(spec.Rows, spec.Cols, spec.Data, cfg.Rank, cfg.OutlierQuantile)
	if err != nil { return nil, err }
	var shards []Shard
	// D shard: shape + fp16
	db := new(bytes.Buffer)
	binary.Write(db, binary.LittleEndian, uint32(spec.Rows))
	binary.Write(db, binary.LittleEndian, uint32(spec.Cols))
	db.Write(fp16bytes(D))
	shards = append(shards, Shard{Type: 3, Scope: spec.Scope, Comp: 0, Data: db.Bytes()})
	// L shard: full L fp16
	lb := new(bytes.Buffer)
	binary.Write(lb, binary.LittleEndian, uint32(spec.Rows))
	binary.Write(lb, binary.LittleEndian, uint32(spec.Cols))
	lb.Write(fp16bytes(L))
	shards = append(shards, Shard{Type: 0, Scope: spec.Scope, Comp: 0, Data: lb.Bytes()})
	// R shard: PQ
	d := 128
	flat := make([]float32, len(R))
	copy(flat, R)
	pad := (d - (len(flat)%d)) % d
	if pad != 0 { flat = append(flat, make([]float32, pad)...)}
	N := len(flat)/d
	data := make([][]float32, N)
	for i := 0; i < N; i++ { data[i] = flat[i*d:(i+1)*d] }
	m := cfg.PQm
	if d % m != 0 { m = d/8 }
	pq := quant.TrainPQ(data, m, cfg.PQk, 25, 1234)
	codes := pq.Encode(data)
	// pack: rows, cols, d, m, k, n, codebooks, codes
	rb := new(bytes.Buffer)
	binary.Write(rb, binary.LittleEndian, uint32(spec.Rows))
	binary.Write(rb, binary.LittleEndian, uint32(spec.Cols))
	binary.Write(rb, binary.LittleEndian, uint16(d))
	binary.Write(rb, binary.LittleEndian, uint16(m))
	binary.Write(rb, binary.LittleEndian, uint16(pq.K))
	binary.Write(rb, binary.LittleEndian, uint32(N))
	// codebooks flattened per subvector
	for i := 0; i < m; i++ { rb.Write(float32SliceToBytes(pq.Codebooks[i])) }
	// codes
	for i := 0; i < N; i++ { rb.Write(codes[i]) }
	shards = append(shards, Shard{Type: 1, Scope: spec.Scope, Comp: 0, Data: rb.Bytes()})
	// S shard: rows, cols, n, idx(i32,i32), vals(f32)
	sb := new(bytes.Buffer)
	binary.Write(sb, binary.LittleEndian, uint32(spec.Rows))
	binary.Write(sb, binary.LittleEndian, uint32(spec.Cols))
	binary.Write(sb, binary.LittleEndian, uint32(len(Sind)))
	for _, ij := range Sind { binary.Write(sb, binary.LittleEndian, ij) }
	sb.Write(float32SliceToBytes(Sval))
	shards = append(shards, Shard{Type: 2, Scope: spec.Scope, Comp: 0, Data: sb.Bytes()})
	return shards, nil
}

func float32SliceToBytes(a []float32) []byte {
	b := new(bytes.Buffer)
	for _, v := range a { binary.Write(b, binary.LittleEndian, v) }
	return b.Bytes()
}
