//go:build cuda

package gpu

import (
    "os"
    "testing"
)

func TestCUDAAvailable(t *testing.T) {
    if !Available() { t.Skip("CUDA not available on this runner") }
    // simple matvec sanity
    rows, cols := 2, 3
    A := []float32{1,2,3,4,5,6}
    x := []float32{0.5, 0.25, -1}
    y := make([]float32, rows)
    if !MatVecF32(y, A, rows, cols, x) { t.Fatalf("MatVecF32 failed") }
}

