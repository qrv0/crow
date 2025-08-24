//go:build cuda

package gpu

/*
#cgo LDFLAGS: -lcublas -lcudart
#include <cuda_runtime.h>
#include <cublas_v2.h>

static const char* cudaErrStr(cudaError_t e) { return cudaGetErrorString(e); }

typedef struct {
    cublasHandle_t handle;
    int ok;
} gpu_ctx;

static gpu_ctx G = {0};

static const char* gpu_init() {
    if (G.ok) return NULL;
    cublasStatus_t st = cublasCreate(&G.handle);
    if (st != CUBLAS_STATUS_SUCCESS) return "cublasCreate failed";
    G.ok = 1;
    return NULL;
}

static void gpu_close() {
    if (G.ok) { cublasDestroy(G.handle); G.ok = 0; }
}

static const char* gpu_matvec_f32(const float* A_rm, int rows, int cols, const float* x, float* y) {
    // Interpret A (row-major rows x cols) as column-major (cols x rows) and use transpose op.
    if (!G.ok) return "not initialized";
    size_t Asz = (size_t)rows * (size_t)cols * sizeof(float);
    size_t xsz = (size_t)cols * sizeof(float);
    size_t ysz = (size_t)rows * sizeof(float);
    float *dA = NULL, *dx = NULL, *dy = NULL;
    cudaError_t ce;
    ce = cudaMalloc((void**)&dA, Asz); if (ce != cudaSuccess) return cudaErrStr(ce);
    ce = cudaMalloc((void**)&dx, xsz); if (ce != cudaSuccess) { cudaFree(dA); return cudaErrStr(ce);} 
    ce = cudaMalloc((void**)&dy, ysz); if (ce != cudaSuccess) { cudaFree(dA); cudaFree(dx); return cudaErrStr(ce);} 
    ce = cudaMemcpy(dA, A_rm, Asz, cudaMemcpyHostToDevice); if (ce != cudaSuccess) { cudaFree(dA); cudaFree(dx); cudaFree(dy); return cudaErrStr(ce);} 
    ce = cudaMemcpy(dx, x,   xsz, cudaMemcpyHostToDevice); if (ce != cudaSuccess) { cudaFree(dA); cudaFree(dx); cudaFree(dy); return cudaErrStr(ce);} 
    // y on device initialized to 0
    ce = cudaMemset(dy, 0, ysz); if (ce != cudaSuccess) { cudaFree(dA); cudaFree(dx); cudaFree(dy); return cudaErrStr(ce);} 
    const float alpha = 1.0f, beta = 1.0f;
    int m = cols, n = rows, lda = cols, incx = 1, incy = 1;
    // dy = alpha*op(A)*dx + beta*dy, with op(A)=A^T (since we pass row-major A)
    cublasStatus_t st = cublasSgemv(G.handle, CUBLAS_OP_T, m, n, &alpha, dA, lda, dx, incx, &beta, dy, incy);
    if (st != CUBLAS_STATUS_SUCCESS) { cudaFree(dA); cudaFree(dx); cudaFree(dy); return "cublasSgemv failed"; }
    ce = cudaMemcpy(y, dy, ysz, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    if (ce != cudaSuccess) return cudaErrStr(ce);
    return NULL;
}
*/
import "C"
import "unsafe"

var available bool

func init() {
    if C.gpu_init() == nil {
        available = true
    }
}

// Available reports whether CUDA/cuBLAS is available (build tag + init ok)
func Available() bool { return available }

// MatVecF32 computes y += A*x, with A row-major (rows x cols), using cuBLAS.
func MatVecF32(y []float32, A []float32, rows, cols int, x []float32) bool {
    if !available { return false }
    if len(A) != rows*cols || len(x) != cols || len(y) != rows { return false }
    // Compute tmp = A*x then add to y
    tmp := make([]float32, rows)
    if err := C.gpu_matvec_f32((*C.float)(unsafe.Pointer(&A[0])), C.int(rows), C.int(cols), (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&tmp[0]))); err != nil {
        return false
    }
    for i := range y { y[i] += tmp[i] }
    return true
}

func Close() { C.gpu_close() }

// RPQMatVecF32 decodes PQ-coded blocks using codebooks and accumulates into y (CPU for now).
// cb is flattened [m][k][dsub], with d = m*dsub. codes has length n*m.
func RPQMatVecF32(y []float32, cb []float32, d, m, k, n int, codes []byte, x []float32) bool {
    if len(y) == 0 || len(cb) == 0 || len(codes) != n*m { return false }
    dsub := d / m
    if len(x) < d { return false }
    cols := len(x)
    for r := 0; r < n; r++ {
        baseFlat := r * d
        for i := 0; i < m; i++ {
            code := int(codes[r*m+i])
            cbStart := (i*k + code) * dsub
            for j := 0; j < dsub; j++ {
                flatIdx := baseFlat + i*dsub + j
                col := flatIdx % cols
                row := flatIdx / cols
                if row < len(y) {
                    y[row] += cb[cbStart+j] * x[col]
                }
            }
        }
    }
    return true
}

// SparseAddF32 applies y[row] += val * x[col] for each triplet (ri,ci,val). CPU for now.
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
