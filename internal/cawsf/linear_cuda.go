//go:build cuda

package cawsf

import "github.com/qrv0/crow/internal/gpu"

func init() {
    gpu_Available    = gpu.Available
    gpu_MatVecF32    = gpu.MatVecF32
    gpu_RPQMatVecF32 = gpu.RPQMatVecF32
    gpu_SparseAddF32 = gpu.SparseAddF32
}

