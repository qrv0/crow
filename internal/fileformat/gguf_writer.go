package fileformat

import (
    "bytes"
    "encoding/binary"
    "errors"
    "io"
    "sort"
)

// Minimal GGUF writer sufficient to serialize tensors with metadata.
// It aims to be compatible with GGUF v3 readers for basic tensor containers.

// GGUF value types (subset)
const (
    ggufTypeUint8   = 0
    ggufTypeInt8    = 1
    ggufTypeUint16  = 2
    ggufTypeInt16   = 3
    ggufTypeUint32  = 4
    ggufTypeInt32   = 5
    ggufTypeFloat32 = 6
    ggufTypeBool    = 7
    ggufTypeString  = 8
    ggufTypeArray   = 9
    ggufTypeUint64  = 10
    ggufTypeInt64   = 11
    ggufTypeFloat64 = 12
)

// GGML tensor type ids (subset)
const (
    ggmlTypeF32 = 0
)

type GGUFKV struct {
    Key   string
    Type  uint32
    Value any
}

// GGUFTensor describes one tensor to serialize.
type GGUFTensor struct {
    Name   string
    Dims   []uint64 // row-major: [n_rows, n_cols] etc
    Type   uint32   // ggml type id; we use f32
    Data   []byte   // raw bytes matching type and dims
}

// GGUFWriter builds a GGUF buffer.
type GGUFWriter struct {
    Version int
    KVs     []GGUFKV
    Tensors []GGUFTensor
    // alignment for tensor data region
    DataAlign int
}

// Export GGUF type ids for external callers
const (
    GGUFTypeUint8   uint32 = ggufTypeUint8
    GGUFTypeInt8    uint32 = ggufTypeInt8
    GGUFTypeUint16  uint32 = ggufTypeUint16
    GGUFTypeInt16   uint32 = ggufTypeInt16
    GGUFTypeUint32  uint32 = ggufTypeUint32
    GGUFTypeInt32   uint32 = ggufTypeInt32
    GGUFTypeFloat32 uint32 = ggufTypeFloat32
    GGUFTypeBool    uint32 = ggufTypeBool
    GGUFTypeString  uint32 = ggufTypeString
    GGUFTypeArray   uint32 = ggufTypeArray
    GGUFTypeUint64  uint32 = ggufTypeUint64
    GGUFTypeInt64   uint32 = ggufTypeInt64
    GGUFTypeFloat64 uint32 = ggufTypeFloat64

    GGMLTypeF32     uint32 = ggmlTypeF32
)

func NewGGUFWriter() *GGUFWriter {
    return &GGUFWriter{Version: 3, DataAlign: 32}
}

func (w *GGUFWriter) AddKV(k GGUFKV) { w.KVs = append(w.KVs, k) }
func (w *GGUFWriter) AddTensor(t GGUFTensor) { w.Tensors = append(w.Tensors, t) }

func alignUp64(x, a uint64) uint64 {
    r := x % a
    if r == 0 { return x }
    return x + (a - r)
}

func writeU64(buf *bytes.Buffer, v uint64) { _ = binary.Write(buf, binary.LittleEndian, v) }
func writeU32(buf *bytes.Buffer, v uint32) { _ = binary.Write(buf, binary.LittleEndian, v) }

func writeString(buf *bytes.Buffer, s string) {
    writeU64(buf, uint64(len(s)))
    buf.WriteString(s)
}

func writeKV(buf *bytes.Buffer, kv GGUFKV) error {
    writeString(buf, kv.Key)
    writeU32(buf, kv.Type)
    switch kv.Type {
    case ggufTypeString:
        str, _ := kv.Value.(string)
        writeString(buf, str)
    case ggufTypeBool:
        b, _ := kv.Value.(bool)
        var u uint8
        if b { u = 1 }
        buf.WriteByte(u)
    case ggufTypeUint8:
        v := kv.Value.(uint8)
        buf.WriteByte(v)
    case ggufTypeUint16:
        v := kv.Value.(uint16)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeUint32:
        v := kv.Value.(uint32)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeUint64:
        v := kv.Value.(uint64)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeInt8:
        v := kv.Value.(int8)
        buf.WriteByte(uint8(v))
    case ggufTypeInt16:
        v := kv.Value.(int16)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeInt32:
        v := kv.Value.(int32)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeInt64:
        v := kv.Value.(int64)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeFloat32:
        v := kv.Value.(float32)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeFloat64:
        v := kv.Value.(float64)
        _ = binary.Write(buf, binary.LittleEndian, v)
    case ggufTypeArray:
        // value for array must be: struct { ElemType uint32; Elems []any }
        arr, ok := kv.Value.(struct{ ElemType uint32; Elems []any })
        if !ok { return errors.New("gguf: array kv value must be struct{ElemType, Elems}") }
        writeU32(buf, arr.ElemType)
        writeU64(buf, uint64(len(arr.Elems)))
        for _, e := range arr.Elems {
            switch arr.ElemType {
            case ggufTypeUint32:
                _ = binary.Write(buf, binary.LittleEndian, e.(uint32))
            case ggufTypeUint64:
                _ = binary.Write(buf, binary.LittleEndian, e.(uint64))
            case ggufTypeInt32:
                _ = binary.Write(buf, binary.LittleEndian, e.(int32))
            case ggufTypeInt64:
                _ = binary.Write(buf, binary.LittleEndian, e.(int64))
            case ggufTypeFloat32:
                _ = binary.Write(buf, binary.LittleEndian, e.(float32))
            case ggufTypeFloat64:
                _ = binary.Write(buf, binary.LittleEndian, e.(float64))
            case ggufTypeBool:
                if e.(bool) { buf.WriteByte(1) } else { buf.WriteByte(0) }
            case ggufTypeString:
                writeString(buf, e.(string))
            default:
                return errors.New("gguf: unsupported array elem type")
            }
        }
    default:
        return errors.New("gguf: unsupported kv type")
    }
    return nil
}

// Write builds the GGUF binary into w. It sorts tensors by name for deterministic layout.
func (w *GGUFWriter) Write(out io.Writer) error {
    // Sort tensor metadata by name for stability
    sort.Slice(w.Tensors, func(i, j int) bool { return w.Tensors[i].Name < w.Tensors[j].Name })
    // 1) Build header
    var head bytes.Buffer
    head.WriteString("GGUF")
    writeU32(&head, uint32(w.Version))
    writeU64(&head, uint64(len(w.Tensors)))
    writeU64(&head, uint64(len(w.KVs)))

    // 2) Build KV block
    var kvb bytes.Buffer
    for _, kv := range w.KVs {
        if err := writeKV(&kvb, kv); err != nil { return err }
    }

    // 3) Build tensor infos with offsets (need to compute layout)
    // First, compute size per tensor and assign offsets in tensor data region with alignment.
    offs := make([]uint64, len(w.Tensors))
    var cur uint64 = 0
    align := uint64(w.DataAlign)
    sizes := make([]uint64, len(w.Tensors))
    for i, t := range w.Tensors {
        // verify data length matches dims * type size
        elemSize := uint64(4) // f32
        expect := elemSize
        for _, d := range t.Dims { expect *= d }
        if uint64(len(t.Data)) != expect {
            return errors.New("gguf: tensor data size mismatch")
        }
        cur = alignUp64(cur, align)
        offs[i] = cur
        sizes[i] = expect
        cur += expect
    }

    var tinf bytes.Buffer
    for i, t := range w.Tensors {
        writeString(&tinf, t.Name)
        writeU32(&tinf, uint32(len(t.Dims)))
        for _, d := range t.Dims { writeU64(&tinf, d) }
        writeU32(&tinf, t.Type)
        writeU64(&tinf, offs[i])
    }

    // 4) Assemble all and then tensor data
    // Header + KV + tensor infos form the metadata block. After that, we align to DataAlign before tensors.
    var meta bytes.Buffer
    meta.Write(head.Bytes())
    meta.Write(kvb.Bytes())
    meta.Write(tinf.Bytes())

    // Compute pad to alignment for data start relative to file start
    pad := int(alignUp64(uint64(meta.Len()), align) - uint64(meta.Len()))
    if pad > 0 { meta.Write(bytes.Repeat([]byte{0}, pad)) }

    // Write meta
    if _, err := out.Write(meta.Bytes()); err != nil { return err }

    // Write tensors in the assigned order with alignment gaps as needed
    // Build a single contiguous region respecting offsets.
    var curPos uint64 = uint64(meta.Len())
    for i, t := range w.Tensors {
        want := offs[i] + uint64(meta.Len())
        if curPos < want {
            gap := int(want - curPos)
            if gap > 0 { if _, err := out.Write(bytes.Repeat([]byte{0}, gap)); err != nil { return err } }
            curPos = want
        }
        if _, err := out.Write(t.Data); err != nil { return err }
        curPos += sizes[i]
    }
    return nil
}
