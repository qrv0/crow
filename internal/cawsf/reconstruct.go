package cawsf

import (
	"encoding/binary"
	"fmt"
	"math"

)

// constants for shard types
const (
	shL = 0
	shR = 1
	shS = 2
	shD = 3
)

type ShardHeader struct {
	Type  uint8  // 0=L,1=R,2=S,3=D
	Scope uint16
	Comp  uint8  // 0=raw
	Usize uint32
	Csize uint32
}

type BankIndex struct {
	Records []BankRec
}

// CodebookPool holds shared PQ codebooks referenced by R shards.
// Layout of CODEBOOKS section (little endian):
// u16 count; then repeated: u16 id; u16 d; u16 m; u16 k; u32 size; bytes (m*k*(d/m)) float32
type CodebookEntry struct {
	ID uint16
	D  int
	M  int
	K  int
	Data []float32 // flattened [m][k][dsub]
}

type CodebookPool struct {
	Entries map[uint16]CodebookEntry
}

func ParseCodebookPool(b []byte) (*CodebookPool, error) {
	pool := &CodebookPool{Entries: make(map[uint16]CodebookEntry)}
	if len(b) == 0 { return pool, nil }
	if len(b) < 2 { return nil, fmt.Errorf("codebooks: short header") }
	cnt := int(binary.LittleEndian.Uint16(b[0:2]))
	off := 2
	for i := 0; i < cnt; i++ {
		if off+12 > len(b) { return nil, fmt.Errorf("codebooks: short entry header") }
		id := binary.LittleEndian.Uint16(b[off:off+2])
		d := int(binary.LittleEndian.Uint16(b[off+2:off+4]))
		m := int(binary.LittleEndian.Uint16(b[off+4:off+6]))
		k := int(binary.LittleEndian.Uint16(b[off+6:off+8]))
		size := int(binary.LittleEndian.Uint32(b[off+8:off+12]))
		off += 12
		if off+size > len(b) { return nil, fmt.Errorf("codebooks: short data") }
		dataBytes := b[off:off+size]
		off += size
		// convert to float32 slice
		if size%4 != 0 { return nil, fmt.Errorf("codebooks: size not multiple of 4") }
		vals := make([]float32, size/4)
		for j := 0; j < len(vals); j++ {
			vals[j] = math.Float32frombits(binary.LittleEndian.Uint32(dataBytes[j*4:]))
		}
		pool.Entries[id] = CodebookEntry{ID: id, D: d, M: m, K: k, Data: vals}
	}
	return pool, nil
}

type BankRec struct {
	Offset int
	Hdr    ShardHeader
}

func IndexShardBank(bank []byte) (*BankIndex, error) {
	var idx BankIndex
	off := 0
	for off+12 <= len(bank) {
		h := ShardHeader{
			Type:  bank[off+0],
			Scope: uint16(bank[off+1]) | uint16(bank[off+2])<<8,
			Comp:  bank[off+3],
			Usize: binary.LittleEndian.Uint32(bank[off+4:off+8]),
			Csize: binary.LittleEndian.Uint32(bank[off+8:off+12]),
		}
		off += 12
		if off+int(h.Csize) > len(bank) { break }
		idx.Records = append(idx.Records, BankRec{Offset: off, Hdr: h})
		off += int(h.Csize)
	}
	return &idx, nil
}

// Reconstruct returns a dense weight matrix for a given scope id.
// It expects shards for that scope: L(fp16+shape), D(fp16+shape), R(PQ payload), S(sparse payload)
func ReconstructForScope(bank []byte, scope uint16) (rows int, cols int, data []float32, err error) {
	return ReconstructForScopeWithPool(bank, nil, scope)
}

func ReconstructForScopeWithPool(bank []byte, pool *CodebookPool, scope uint16) (rows int, cols int, data []float32, err error) {
	idx, err := IndexShardBank(bank)
	if err != nil { return }
	var L, D []float32
	var R [][]float32
	var shapeRows, shapeCols int
	var Sind [][2]int32
	var Sval []float32
	for _, rec := range idx.Records {
		if rec.Hdr.Scope != scope { continue }
		payloadRaw := bank[rec.Offset: rec.Offset+int(rec.Hdr.Csize)]
		payload, derr := decompressShard(rec.Hdr.Comp, payloadRaw)
		if derr != nil { return 0,0,nil, derr }
		switch rec.Hdr.Type {
		case 0: // L
			rows, cols, mat, e := readFP16WithShape(payload)
			if e != nil { return 0,0,nil,e }
			shapeRows, shapeCols = rows, cols
			L = mat
		case 3: // D
			rows, cols, mat, e := readFP16WithShape(payload)
			if e != nil { return 0,0,nil,e }
			shapeRows, shapeCols = rows, cols
			D = mat
		case 1: // R PQ
			rows, cols, mat, e := decodeR(payload)
			if e != nil {
				// try shared codebook path if pool provided
				if pool != nil {
					mat2, e2 := decodeRWithPool(rows, cols, payload, pool)
					if e2 != nil { return 0,0,nil,e2 }
					shapeRows, shapeCols = rows, cols
					R = append(R, mat2)
					break
				}
				return 0,0,nil,e
			}
			shapeRows, shapeCols = rows, cols
			R = append(R, mat)
		case 2: // S
			r, c, sind, sval, e := decodeS(payload)
			if e != nil { return 0,0,nil,e }
			shapeRows, shapeCols = r, c
			Sind, Sval = sind, sval
		}
	}
	if shapeRows == 0 || shapeCols == 0 { return 0,0,nil, fmt.Errorf("scope %d not found", scope) }
	data = make([]float32, shapeRows*shapeCols)
	if L != nil { addInPlace(data, L) }
	if D != nil { addInPlace(data, D) }
	for _, rr := range R { addInPlace(data, rr) }
	if len(Sind) > 0 {
		for i, ij := range Sind {
			r := int(ij[0]); c := int(ij[1])
			data[r*shapeCols + c] += Sval[i]
		}
	}
	return shapeRows, shapeCols, data, nil
}

func addInPlace(dst, src []float32) {
	for i := range src { dst[i] += src[i] }
}

func readFP16WithShape(p []byte) (rows, cols int, out []float32, err error) {
	if len(p) < 8 { return 0,0,nil, fmt.Errorf("short fp16 payload") }
	rows = int(binary.LittleEndian.Uint32(p[0:4]))
	cols = int(binary.LittleEndian.Uint32(p[4:8]))
	n := rows*cols
	out = make([]float32, n)
	off := 8
	for i := 0; i < n; i++ {
		if off+2 > len(p) { return 0,0,nil, fmt.Errorf("short fp16 data") }
		h := uint16(p[off]) | uint16(p[off+1])<<8
		off += 2
		out[i] = fp16to32(h)
	}
	return
}

func fp16to32(h uint16) float32 {
	s := uint32(h>>15) & 0x1
	e := uint32(h>>10) & 0x1F
	m := uint32(h) & 0x3FF
	var f uint32
	if e == 0 {
		if m == 0 { f = s << 31 } else {
			// subnormal -> normalize approximately
			e2 := uint32(127 - 15 + 1)
			m2 := uint32(m) << 13
			for (m2 & (1<<23)) == 0 { m2 <<= 1; e2-- }
			m2 &= (1<<23)-1
	f = (s<<31) | (e2<<23) | m2
	}
	} else if e == 0x1F {
		f = (s<<31) | (0xFF<<23) | (m<<13)
	} else {
		e2 := (e) - 15 + 127
		f = (s<<31) | (e2<<23) | (m<<13)
	}
	return math.Float32frombits(f)
}

func decodeR(p []byte) (rows, cols int, mat []float32, err error) {
	// Two possible layouts:
	// A) Embedded codebooks: rows:u32, cols:u32, d:u16, m:u16, k:u16, n:u32, cb:(m*k*dsub*f32), codes:(n*m*u8)
	// B) Shared codebooks:   rows:u32, cols:u32, d:u16, m:u16, k:u16, n:u32, cb_id:u16, codes:(n*m*u8)

	// Two possible layouts:
	// A) Embedded codebooks: rows:u32, cols:u32, d:u16, m:u16, k:u16, n:u32, cb:(m*k*dsub*f32), codes:(n*m*u8)
	// B) Shared codebooks:   rows:u32, cols:u32, d:u16, m:u16, k:u16, n:u32, cb_id:u16, codes:(n*m*u8)

	if len(p) < 8+8 { return 0,0,nil, fmt.Errorf("short R payload") }
	rows = int(binary.LittleEndian.Uint32(p[0:4]))
	cols = int(binary.LittleEndian.Uint32(p[4:8]))
	d := int(binary.LittleEndian.Uint16(p[8:10]))
	m := int(binary.LittleEndian.Uint16(p[10:12]))
	k := int(binary.LittleEndian.Uint16(p[12:14]))
	n := int(binary.LittleEndian.Uint32(p[14:18]))
	dsub := d/m
	// Check if shared codebook layout by seeing if remaining length matches (2 + n*m)
	if len(p) >= 18+2 && (len(p)-(18+2)) == n*m {
		cbID := int(binary.LittleEndian.Uint16(p[18:20]))
		_ = p[20:]
		// Without external codebook pool here, signal to caller to use pool path by returning error
		return rows, cols, nil, fmt.Errorf("R shard references shared codebook id=%d: external codebooks required", cbID)
	}
	// Embedded codebooks fallback
	cbSize := m*k*dsub*4
	if 18+cbSize > len(p) { return 0,0,nil, fmt.Errorf("short codebooks") }
	cb := p[18:18+cbSize]
	codes := p[18+cbSize:]
	if len(codes) != n*m { return 0,0,nil, fmt.Errorf("codes size mismatch") }
	blocks := make([]float32, n*d)
	// decode per sub-vector
	for i := 0; i < m; i++ {
		for r := 0; r < n; r++ {
			idx := int(codes[r*m + i])
			start := (i*k + idx)*dsub*4
			for j := 0; j < dsub; j++ {
				off := start + j*4
				blocks[r*d + i*dsub + j] = math.Float32frombits(binary.LittleEndian.Uint32(cb[off:off+4]))
			}
		}
	}
	flat := blocks
	need := rows*cols
	flat = flat[:min(need, len(flat))]
	mat = make([]float32, need)
	copy(mat, flat)
	return
}

func decodeS(p []byte) (rows, cols int, idx [][2]int32, vals []float32, err error) {
	if len(p) < 12 { return 0,0,nil,nil, fmt.Errorf("short S payload") }
	rows = int(binary.LittleEndian.Uint32(p[0:4]))
	cols = int(binary.LittleEndian.Uint32(p[4:8]))
	n := int(binary.LittleEndian.Uint32(p[8:12]))
	pos := 12
	idx = make([][2]int32, n)
	for i := 0; i < n; i++ {
		if pos+8 > len(p) { return 0,0,nil,nil, fmt.Errorf("short S indices") }
		idx[i][0] = int32(binary.LittleEndian.Uint32(p[pos:pos+4]))
		idx[i][1] = int32(binary.LittleEndian.Uint32(p[pos+4:pos+8]))
		pos += 8
	}
	vals = make([]float32, n)
	for i := 0; i < n; i++ {
		if pos+4 > len(p) { return 0,0,nil,nil, fmt.Errorf("short S values") }
		vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[pos:pos+4]))
		pos += 4
	}
	return
}

func decodeRWithPool(rows, cols int, p []byte, pool *CodebookPool) ([]float32, error) {
	if len(p) < 20 { return nil, fmt.Errorf("short R payload (pool)") }
	d := int(binary.LittleEndian.Uint16(p[8:10]))
	m := int(binary.LittleEndian.Uint16(p[10:12]))
	k := int(binary.LittleEndian.Uint16(p[12:14]))
	n := int(binary.LittleEndian.Uint32(p[14:18]))
	cbID := binary.LittleEndian.Uint16(p[18:20])
	codes := p[20:]
	if len(codes) != n*m { return nil, fmt.Errorf("codes size mismatch") }
	entry, ok := pool.Entries[cbID]
	if !ok { return nil, fmt.Errorf("codebook id %d not found", cbID) }
	if entry.M != 0 && entry.M != m { return nil, fmt.Errorf("codebook m mismatch: %d vs %d", entry.M, m) }
	if entry.K != 0 && entry.K != k { return nil, fmt.Errorf("codebook k mismatch: %d vs %d", entry.K, k) }
	// derive dsub from entry or header
	dsub := d / m
	if entry.D != 0 && entry.D/m != dsub { return nil, fmt.Errorf("codebook d mismatch") }
	// decode
	blocks := make([]float32, n*d)
	for i := 0; i < m; i++ {
		for r := 0; r < n; r++ {
			idx := int(codes[r*m + i])
			start := (i*k + idx)*dsub
			for j := 0; j < dsub; j++ {
				blocks[r*d + i*dsub + j] = entry.Data[start + j]
			}
		}
	}
	need := rows*cols
	mat := make([]float32, need)
	copy(mat, blocks[:min(need, len(blocks))])
	return mat, nil
}

func min(a,b int) int { if a<b {return a}; return b }
