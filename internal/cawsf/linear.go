package cawsf

import (
    "encoding/binary"
    "fmt"
    "math"
    "os"
)

// lightweight indirection to internal/gpu to keep this file building without cuda tag
var (
    gpu_Available    = func() bool { return false }
    gpu_MatVecF32    = func(y []float32, A []float32, rows, cols int, x []float32) bool { return false }
    gpu_RPQMatVecF32 = func(y []float32, cb []float32, d, m, k, n int, codes []byte, x []float32) bool { return false }
    gpu_SparseAddF32 = func(y []float32, rows, cols int, ri []int32, ci []int32, val []float32, x []float32) bool { return false }
)

// MultiplyScopeWithPool computes y = W*x for the given scope using shards in bank and an optional codebook pool.
// - L and D shards are applied as dense matvec.
// - R shards (PQ) are applied by decoding blocks on the fly (no full materialization).
// - S shards (sparse) add their contributions.
// x must have length = cols. Returns y with length = rows.
func MultiplyScopeWithPool(bank []byte, pool *CodebookPool, scope uint16, x []float32) ([]float32, int, int, error) {
	idx, err := IndexShardBank(bank)
	if err != nil { return nil,0,0, err }
	var rows, cols int
	var y []float32
	var haveShape bool
	// First pass: get shape from any shard of scope
	for _, rec := range idx.Records {
		if rec.Hdr.Scope != scope { continue }
		payload := bank[rec.Offset: rec.Offset+int(rec.Hdr.Csize)]
		switch rec.Hdr.Type {
		case shL, shD:
			if len(payload) < 8 { return nil,0,0, fmt.Errorf("short payload") }
			r := int(binary.LittleEndian.Uint32(payload[0:4]))
			c := int(binary.LittleEndian.Uint32(payload[4:8]))
			rows, cols = r, c; haveShape = true
		case shR:
			if len(payload) < 8+8 { return nil,0,0, fmt.Errorf("short R payload") }
			r := int(binary.LittleEndian.Uint32(payload[0:4]))
			c := int(binary.LittleEndian.Uint32(payload[4:8]))
			rows, cols = r, c; haveShape = true
		case shS:
			if len(payload) < 12 { return nil,0,0, fmt.Errorf("short S payload") }
			r := int(binary.LittleEndian.Uint32(payload[0:4]))
			c := int(binary.LittleEndian.Uint32(payload[4:8]))
			rows, cols = r, c; haveShape = true
		}
	}
	if !haveShape { return nil,0,0, fmt.Errorf("scope %d not found", scope) }
	if len(x) != cols { return nil,0,0, fmt.Errorf("input length %d != cols %d", len(x), cols) }
	y = make([]float32, rows)
    // Optional CUDA acceleration for L/D shards if enabled
    useCUDA := os.Getenv("CROW_CUDA") == "1"
    lHandled, dHandled := false, false
    if useCUDA {
        // Collect L/D, decode to f32, and apply with GPU matvec
        var Lf, Df []float32
        for _, rec := range idx.Records {
            if rec.Hdr.Scope != scope { continue }
            if rec.Hdr.Type != shL && rec.Hdr.Type != shD { continue }
            payload := bank[rec.Offset: rec.Offset+int(rec.Hdr.Csize)]
            r2, c2, mat, e := readFP16WithShape(payload)
            if e != nil { continue }
            if r2 != rows || c2 != cols { continue }
            if rec.Hdr.Type == shL { Lf = mat }
            if rec.Hdr.Type == shD { Df = mat }
        }
        // Try GPU path
        if len(Lf) == rows*cols || len(Df) == rows*cols {
            // Defer import to avoid dependency when not used
            if gpuMatVec(y, rows, cols, Lf, Df, x) {
                if len(Lf) == rows*cols { lHandled = true }
                if len(Df) == rows*cols { dHandled = true }
            }
        }
    }
    // Apply each shard
    for _, rec := range idx.Records {
        if rec.Hdr.Scope != scope { continue }
        payload := bank[rec.Offset: rec.Offset+int(rec.Hdr.Csize)]
        switch rec.Hdr.Type {
        case shL:
            if lHandled { continue }
            // payload: rows, cols, fp16[rows*cols]
            matvecFP16Add(y, rows, cols, payload[8:], x)
        case shD:
            if dHandled { continue }
            // payload: rows, cols, fp16[rows*cols]
            matvecFP16Add(y, rows, cols, payload[8:], x)
        case shR:
            if err := applyRAddOptimized(y, rows, cols, payload, x, pool); err != nil { return nil,0,0, err }
        case shS:
            applySAddOptimized(y, rows, cols, payload, x)
        }
    }
    return y, rows, cols, nil
}

// gpuMatVec tries CUDA path via internal/gpu; returns true if used
func gpuMatVec(y []float32, rows, cols int, Lf, Df, x []float32) bool {
    used := false
    // Use reflection-free import pattern by referencing package here
    // Note: this file always compiles; gpu.Available() is false without cuda tag
    if gpuAvailable() {
        if len(Lf) == rows*cols {
            if gpuMat(y, Lf, rows, cols, x) { used = true }
        }
        if len(Df) == rows*cols {
            if gpuMat(y, Df, rows, cols, x) { used = true }
        }
    }
    return used
}

// The following tiny wrappers let us call into internal/gpu without import cycles in non-cuda builds.
func gpuAvailable() bool {
    // shadow import to avoid unused in non-cuda builds; actual linkage via build tags
    return gpu_Available()
}

func gpuMat(y, A []float32, rows, cols int, x []float32) bool { return gpu_MatVecF32(y, A, rows, cols, x) }

func matvecFP16Add(y []float32, rows, cols int, data []byte, x []float32) {
	off := 0
	for i := 0; i < rows; i++ {
		s := float32(0)
		for j := 0; j < cols; j++ {
			if off+2 > len(data) { return }
			h := uint16(data[off]) | uint16(data[off+1])<<8
			off += 2
			w := fp16to32(h)
			s += w * x[j]
		}
		y[i] += s
	}
}

func applySAddOptimized(y []float32, rows, cols int, payload []byte, x []float32) {
	if len(payload) < 12 { return }
	n := int(binary.LittleEndian.Uint32(payload[8:12]))
	// Try GPU-assisted path by parsing indices and values once
	ri := make([]int32, n)
	ci := make([]int32, n)
	val := make([]float32, n)
	off := 12
	for i := 0; i < n; i++ {
		if off+8 > len(payload) { return }
		ri[i] = int32(binary.LittleEndian.Uint32(payload[off:off+4]))
		ci[i] = int32(binary.LittleEndian.Uint32(payload[off+4:off+8]))
		off += 8
		if off+4 > len(payload) { return }
		val[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[off:off+4]))
		off += 4
	}
	if gpu_SparseAddF32(y, rows, cols, ri, ci, val, x) { return }
	// Fallback CPU inline
	for i := 0; i < n; i++ {
		r := int(ri[i]); c := int(ci[i])
		if r < 0 || r >= rows || c < 0 || c >= cols { continue }
		y[r] += val[i] * x[c]
	}
}

func applyRAddOptimized(y []float32, rows, cols int, payload []byte, x []float32, pool *CodebookPool) error {
	if len(payload) < 18 { return fmt.Errorf("short R payload") }
	d := int(binary.LittleEndian.Uint16(payload[8:10]))
	m := int(binary.LittleEndian.Uint16(payload[10:12]))
	k := int(binary.LittleEndian.Uint16(payload[12:14]))
	n := int(binary.LittleEndian.Uint32(payload[14:18]))
	dsub := d / m
	// shared codebooks case: try GPU/CPU-accelerated path first
	if len(payload) >= 20 && (len(payload)-(18+2)) == n*m {
        if pool != nil {
            cbID := binary.LittleEndian.Uint16(payload[18:20])
            entry, ok := pool.Entries[cbID]
            if ok {
                // Use gpu RPQ path if available; fallback to CPU path via gpu_* wrappers
                if gpu_RPQMatVecF32(y, entry.Data, d, m, k, n, payload[20:], x) {
                    return nil
                }
            }
        }
		if pool == nil { return fmt.Errorf("shared codebooks referenced but pool is nil") }
		cbID := binary.LittleEndian.Uint16(payload[18:20])
		entry, ok := pool.Entries[cbID]
		if !ok { return fmt.Errorf("codebook id %d not found", cbID) }
		codes := payload[20:]
		// blocks decode loop
		for r := 0; r < n; r++ {
			// decode one block vector of length d into tmp
			// tmp constructed from codebooks using codes[r*m+i]
			startFlat := r * d
			for i := 0; i < m; i++ {
				idx := int(codes[r*m + i])
				base := (i*k + idx) * dsub
				for j := 0; j < dsub; j++ {
					flatIdx := startFlat + i*dsub + j
					row := flatIdx / cols
					col := flatIdx % cols
					if row < rows {
						y[row] += entry.Data[base+j] * x[col]
					}
				}
			}
		}
		return nil
	}
	// embedded codebooks
	cbSize := m*k*dsub*4
	if 18+cbSize > len(payload) { return fmt.Errorf("short codebooks") }
	cb := payload[18 : 18+cbSize]
	codes := payload[18+cbSize:]
	if len(codes) != n*m { return fmt.Errorf("codes size mismatch") }
	for r := 0; r < n; r++ {
		startFlat := r * d
		for i := 0; i < m; i++ {
			idx := int(codes[r*m + i])
			base := (i*k + idx) * dsub * 4
			for j := 0; j < dsub; j++ {
				off := base + j*4
				v := math.Float32frombits(binary.LittleEndian.Uint32(cb[off:off+4]))
				flatIdx := startFlat + i*dsub + j
				row := flatIdx / cols
				col := flatIdx % cols
				if row < rows {
					y[row] += v * x[col]
				}
			}
		}
	}
	return nil
}
