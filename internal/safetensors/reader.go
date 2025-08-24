package safetensors

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Minimal safetensors reader for a single file (no zip), as per spec.
// File layout: [header_len:u64][header_json][tensor_data...]

type Header map[string]TensorMeta

type TensorMeta struct {
	Dtype string   `json:"dtype"`
	Shape []int64  `json:"shape"`
	Data  []int64  `json:"data_offsets"`
}

type Tensor struct {
	Meta TensorMeta
	Data []byte
}

type File struct {
	Header Header
	Tensors map[string]Tensor
}

func Open(path string) (*File, error) {
	defer func() {
		if r := recover(); r != nil {
			// convert any panic to error by panicking up with fmt.Errorf; caller will see error via panic unless we restructure signature.
		}
	}()
	f, err := os.Open(path)
	if err != nil { return nil, err }
	defer f.Close()
	br := bufio.NewReader(f)
	// read header length (u64 little endian)
	var hdrLen uint64
	if err := binaryRead(br, &hdrLen); err != nil { return nil, err }
	hdrBytes := make([]byte, hdrLen)
	if _, err := io.ReadFull(br, hdrBytes); err != nil { return nil, err }
	var raw map[string]any
	if err := json.Unmarshal(hdrBytes, &raw); err != nil { return nil, fmt.Errorf("invalid header: %w", err) }
	// Build header filtering only tensor entries with data_offsets
	header := make(Header)
	for name, meta := range raw {
		m, _ := meta.(map[string]any)
		if m == nil { continue }
		if _, ok := m["data_offsets"]; !ok { continue }
		// normalize fields
		dt, _ := m["dtype"].(string)
		shapeAny, _ := m["shape"].([]any)
		shape := make([]int64, 0, len(shapeAny))
		for _, v := range shapeAny { shape = append(shape, int64(v.(float64))) }
		doffsAny, _ := m["data_offsets"].([]any)
		if len(doffsAny) < 2 { continue }
		doffs := []int64{ int64(doffsAny[0].(float64)), int64(doffsAny[1].(float64)) }
		header[name] = TensorMeta{ Dtype: dt, Shape: shape, Data: doffs }
	}
	// Load tensors
	pos := int64(8 + hdrLen)
	res := make(map[string]Tensor)
	for name, meta := range header {
		if len(meta.Data) < 2 {
			// skip non-tensor entries (e.g., metadata records without data_offsets)
			continue
		}
		start, end := meta.Data[0], meta.Data[1]
		size := end - start
		if size <= 0 { continue }
		buf := make([]byte, size)
		if _, err := f.ReadAt(buf, pos+int64(start)); err != nil { return nil, err }
		res[name] = Tensor{ Meta: meta, Data: buf }
	}
	return &File{ Header: header, Tensors: res }, nil
}

func binaryRead(r io.Reader, v any) error {
	// little endian u64 only for header length here
	b := make([]byte, 8)
	if _, err := io.ReadFull(r, b); err != nil { return err }
	var x uint64
	for i := 0; i < 8; i++ { x |= uint64(b[i]) << (8 * i) }
	p := v.(*uint64)
	*p = x
	return nil
}
