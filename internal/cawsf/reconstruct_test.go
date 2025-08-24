package cawsf

import (
    "bytes"
    "encoding/binary"
    "math"
    "testing"
)

// helper to build fp16 payload with shape
func f16Payload(rows, cols int, f32 []float32) []byte {
    buf := new(bytes.Buffer)
    binary.Write(buf, binary.LittleEndian, uint32(rows))
    binary.Write(buf, binary.LittleEndian, uint32(cols))
    for _, v := range f32 { buf.Write(fp32to16(v)) }
    return buf.Bytes()
}

// local helper: same as convert.fp32to16
func fp32to16(f float32) []byte {
    u := math.Float32bits(f)
    sign := (u >> 31) & 1
    exp := (u >> 23) & 0xFF
    mant := u & 0x7FFFFF
    var h uint16
    if exp == 0 {
        h = uint16(sign << 15)
    } else if exp == 0xFF {
        h = uint16((sign<<15) | (0x1F<<10))
    } else {
        e := int(exp) - 127 + 15
        if e <= 0 { h = uint16(sign<<15) } else if e >= 0x1F { h = uint16((sign<<15)|(0x1F<<10)) } else {
            h = uint16((sign<<15) | (uint32(e)&0x1F)<<10 | ((mant>>13)&0x3FF))
        }
    }
    return []byte{byte(h), byte(h >> 8)}
}

func pack(t uint8, scope uint16, payload []byte) []byte {
    var hdr [12]byte
    hdr[0] = t
    hdr[1] = byte(scope)
    hdr[2] = byte(scope >> 8)
    hdr[3] = 0
    binary.LittleEndian.PutUint32(hdr[4:], uint32(len(payload)))
    binary.LittleEndian.PutUint32(hdr[8:], uint32(len(payload)))
    return append(hdr[:], payload...)
}

func TestReconstructSmall_L_D_S(t *testing.T) {
    // Build a tiny 2x3 matrix W = L + D + S (no R), then reconstruct
    rows, cols := 2, 3
    L := []float32{1,2,3,4,5,6} // row-major 2x3
    D := []float32{0.5,0,0,0,0.25,0}
    Sind := [][2]int32{{0,1},{1,2}}
    Sval := []float32{0.1, -0.2}
    // bank bytes
    var bank []byte
    bank = append(bank, pack(shL, 0, f16Payload(rows, cols, L))...)
    bank = append(bank, pack(shD, 0, f16Payload(rows, cols, D))...)
    // S payload: rows, cols, n, idx(i32,i32), vals(f32)
    sb := new(bytes.Buffer)
    binary.Write(sb, binary.LittleEndian, uint32(rows))
    binary.Write(sb, binary.LittleEndian, uint32(cols))
    binary.Write(sb, binary.LittleEndian, uint32(len(Sind)))
    for _, ij := range Sind { binary.Write(sb, binary.LittleEndian, ij) }
    for _, v := range Sval { binary.Write(sb, binary.LittleEndian, math.Float32bits(v)) }
    bank = append(bank, pack(shS, 0, sb.Bytes())...)

    r, c, mat, err := ReconstructForScope(bank, 0)
    if err != nil { t.Fatalf("reconstruct error: %v", err) }
    if r!=rows || c!=cols { t.Fatalf("shape mismatch") }
    // spot check a few entries
    want00 := L[0] + D[0] // + S at (0,1) only
    if absf(mat[0]-want00) > 1e-2 { t.Fatalf("mat[0] got %f want %f", mat[0], want00) }
    want01 := L[1] + D[1] + 0.1
    if absf(mat[1]-want01) > 1e-2 { t.Fatalf("mat[1] got %f want %f", mat[1], want01) }
}

func TestDecodeRWithPool(t *testing.T) {
    // R payload with shared codebook id=0; simple case m=1, k=2, d=2, n=2
    rows, cols := 2, 2
    d, m, k, n := 2, 1, 2, 2
    // codebook: two vectors [1,0] and [0,1]
    pool := &CodebookPool{Entries: map[uint16]CodebookEntry{0:{ID:0, D:d, M:m, K:k, Data: []float32{1,0,0,1}}}}
    // codes: select [1,0] then [0,1]
    pb := new(bytes.Buffer)
    binary.Write(pb, binary.LittleEndian, uint32(rows))
    binary.Write(pb, binary.LittleEndian, uint32(cols))
    binary.Write(pb, binary.LittleEndian, uint16(d))
    binary.Write(pb, binary.LittleEndian, uint16(m))
    binary.Write(pb, binary.LittleEndian, uint16(k))
    binary.Write(pb, binary.LittleEndian, uint32(n))
    binary.Write(pb, binary.LittleEndian, uint16(0)) // cb id
    pb.Write([]byte{0,1}) // codes for two blocks
    got, err := decodeRWithPool(rows, cols, pb.Bytes(), pool)
    if err != nil { t.Fatalf("decodeRWithPool err: %v", err) }
    if len(got) != rows*cols { t.Fatalf("len got %d", len(got)) }
}

func absf(x float32) float32 { if x<0 { return -x }; return x }

