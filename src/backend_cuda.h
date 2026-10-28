// ============================================================================
// backend_cuda.h - The GPU side of the tensor library.
//
// Every function takes raw device pointers and plain sizes, so this header
// has no CUDA types and can be included anywhere. Two implementations exist:
//   backend_cuda.cu        the real one, compiled by nvcc with `make CUDA=1`
//   backend_cuda_stub.cpp  used otherwise: available() is false and every
//                          other call throws
// tensor.cpp dispatches here when a Tensor lives on the GPU (Device::CUDA).
// Activations are float32 on the device; model weights may be bf16 (uint16
// bit patterns) and are expanded inside the GEMM (cuBLAS bf16 tensor cores).
// All work goes through one stream in order, so calls are synchronous with
// respect to each other; download() waits for the GPU.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace sd {
namespace cuda {

// True if the binary was built with CUDA and a device is present.
bool available();
// Name of device 0 and its memory in bytes (for logs); "" / 0 if unavailable.
const char* device_name();
size_t device_memory();

// Memory. malloc/free go through a small caching allocator.
void* malloc(size_t bytes);
void free(void* p);
void upload(void* dst, const void* src, size_t bytes);    // host -> device
void download(void* dst, const void* src, size_t bytes);  // device -> host (synchronises)
void copy(void* dst, const void* src, size_t bytes);      // device -> device
void zero(void* dst, size_t bytes);
void synchronize();

// GEMM: C[M,N] = A[M,K] * B[N,K]^T (+ bias[N]); row-major, leading dims in elements.
void gemm_nt_f32(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
                 int64_t ldc, const float* bias, bool accumulate);
// Same with bf16 B (weights); A is converted to bf16 on the device for the tensor-core path.
void gemm_nt_bf16(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const uint16_t* B, int64_t ldb,
                  float* C, int64_t ldc, const float* bias, bool accumulate);

// Layers (shapes as in tensor.h). W for conv2d is f32 [Cout, Cin, kh, kw].
void conv2d(const float* x, int64_t N, int64_t Cin, int64_t H, int64_t W, const float* w, int64_t Cout, int64_t kh,
            int64_t kw, const float* bias, int stride, int pad, float* out, int64_t Ho, int64_t Wo);
void group_norm(const float* x, float* y, int64_t N, int64_t C, int64_t HW, int groups, const float* gamma,
                const float* beta, float eps);
void layer_norm(const float* x, float* y, int64_t rows, int64_t C, const float* gamma, const float* beta, float eps);
void rms_norm(const float* x, float* y, int64_t rows, int64_t C, const float* gamma, float eps);
void rms_norm_heads(float* x, int64_t groups, int64_t dim, const float* gamma, float eps);
// x [N, S, heads*dim]; cos/sin [S, dim]
void rope(float* x, int64_t N, int64_t S, int64_t inner, int64_t dim, const float* cos, const float* sin,
          bool interleaved);
void softmax_rows(float* x, int64_t rows, int64_t cols, float scale, bool causal);
// q [N, Sq, heads*d]; k, v [N, Skv, kv_heads*d]; out [N, Sq, heads*d]
void attention(const float* q, const float* k, const float* v, float* out, int64_t N, int64_t Sq, int64_t Skv,
               int heads, int kv_heads, int64_t d, bool causal);
void geglu(const float* x, float* y, int64_t rows, int64_t D);  // x [rows, 2D] -> y [rows, D]
void swiglu(const float* gate, const float* up, float* y, int64_t n);

// Elementwise and shape ops.
void silu(float* x, int64_t n);
void gelu(float* x, int64_t n);
void quick_gelu(float* x, int64_t n);
void add(float* a, const float* b, int64_t n);
void scale(float* x, float s, int64_t n);
void affine(float* x, float a, float b, int64_t n);                       // x = x * a + b
void mul_rows(float* x, const float* v, int64_t rows, int64_t dim);       // x[r, :] *= v[:]
void add_rows_scaled(float* x, const float* y, const float* g, int64_t rows, int64_t dim);  // x[r,:] += g[:] * y[r,:]
void add_channel_bias(float* x, const float* b, int64_t N, int64_t C, int64_t HW);          // x[n,c,:] += b[n,c]
void upsample_nearest2x(const float* x, float* y, int64_t NC, int64_t H, int64_t W);
void concat_channels(const float* a, const float* b, float* y, int64_t N, int64_t Ca, int64_t Cb, int64_t HW);
void nchw_to_nlc(const float* x, float* y, int64_t N, int64_t C, int64_t HW);
void nlc_to_nchw(const float* x, float* y, int64_t N, int64_t C, int64_t HW);
// [C, H, W] <-> [(H/p)*(W/p), p*p*C] with features ordered (ph, pw, c)
void patchify(const float* x, float* y, int64_t C, int64_t H, int64_t W, int p);
void unpatchify(const float* x, float* y, int64_t C, int64_t H, int64_t W, int p);

}  // namespace cuda
}  // namespace sd
