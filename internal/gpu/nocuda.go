//go:build !cuda

package gpu

// CPU fallback (no CUDA build tag)

func Available() bool { return false }

func MatVecF32(y []float32, A []float32, rows, cols int, x []float32) bool {
    // Not available in this build
    return false
}

// CPU implementation of RPQMatVecF32 so non-CUDA builds are fully functional for R shards
func RPQMatVecF32(y []float32, cb []float32, d, m, k, n int, codes []byte, x []float32) bool {
    if len(y) == 0 || len(cb) == 0 || len(codes) != n*m { return false }
    dsub := d / m
    if len(x) < d { return false }
    for r := 0; r < n; r++ {
        baseFlat := r * d
        for i := 0; i < m; i++ {
            code := int(codes[r*m+i])
            cbStart := (i*k + code) * dsub
            for j := 0; j < dsub; j++ {
                flatIdx := baseFlat + i*dsub + j
                col := flatIdx % len(x)
                row := flatIdx / len(x)
                if row < len(y) {
                    y[row] += cb[cbStart+j] * x[col]
                }
            }
        }
    }
    return true
}
// CPU implementation for SparseAddF32 so non-CUDA builds apply S shards
func SparseAddF32(y []float32, rows, cols int, ri []int32, ci []int32, val []float32, x []float32) bool {
    if len(ri) != len(ci) || len(ci) != len(val) { return false }
    for i := range ri {
        r := int(ri[i])
        c := int(ci[i])
        if r >= 0 && r < len(y) && c >= 0 && c < len(x) {
            y[r] += val[i] * x[c]
        }
    }
    return true
}
func Close() {}

