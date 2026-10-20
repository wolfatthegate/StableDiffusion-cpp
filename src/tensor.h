// Minimal dense float32 tensor + the handful of ops Stable Diffusion needs.
// Layouts: activations are NCHW for convolutions and [N, L, C] for attention.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sd {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> s) : shape(std::move(s)), data(numel_of(shape), 0.0f) {}

    static int64_t numel_of(const std::vector<int64_t>& s);
    int64_t numel() const { return (int64_t)data.size(); }
    int ndim() const { return (int)shape.size(); }
    int64_t dim(int i) const { return shape[i < 0 ? i + ndim() : i]; }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    bool empty() const { return data.empty(); }
    Tensor& reshape(std::vector<int64_t> s);
    std::string shape_str() const;
};

// ---- threading -------------------------------------------------------------
void set_num_threads(int n);
int num_threads();
// Runs fn(i) for i in [0, n) on the thread pool (the calling thread helps).
void parallel_for(int64_t n, const std::function<void(int64_t)>& fn);
// Splits [0, n) into contiguous ranges of at least min_chunk and runs fn(begin, end).
void parallel_chunks(int64_t n, int64_t min_chunk, const std::function<void(int64_t, int64_t)>& fn);

// ---- GEMM ------------------------------------------------------------------
// C[M,N] = A[M,K] * B[N,K]^T (+ bias[N]).  If accumulate, the product is added to C.
// Both operands are "row-major with K contiguous", which is the natural layout for
// y = x W^T (PyTorch Linear), im2col convolutions and Q K^T attention scores.
void gemm_nt(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb,
             float* C, int64_t ldc, const float* bias = nullptr, bool accumulate = false);

// ---- layers ----------------------------------------------------------------
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b);           // x[..., in] -> [..., out]
Tensor conv2d(const Tensor& x, const Tensor& W, const Tensor* b, int stride, int pad);  // NCHW
Tensor group_norm(const Tensor& x, int groups, const Tensor& gamma, const Tensor& beta, float eps);
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps);  // over last dim
// q [N,Sq,inner], k/v [N,Skv,inner] -> [N,Sq,inner]; softmax(q k^T / sqrt(d)) v per head
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, bool causal = false);
Tensor geglu(const Tensor& x);  // [..., 2D] -> [..., D]: a * gelu(g)

// ---- elementwise / shape ---------------------------------------------------
void silu_(Tensor& x);
void gelu_(Tensor& x);
void quick_gelu_(Tensor& x);
void add_(Tensor& a, const Tensor& b);  // a += b (same shape)
void scale_(Tensor& x, float s);
void softmax_rows_(float* x, int64_t rows, int64_t cols, float scale, bool causal);
Tensor upsample_nearest2x(const Tensor& x);
Tensor concat_channels(const Tensor& a, const Tensor& b);  // NCHW along C
Tensor concat_batch(const Tensor& a, const Tensor& b);     // along dim 0
Tensor nchw_to_nlc(const Tensor& x);                       // [N,C,H,W] -> [N,H*W,C]
Tensor nlc_to_nchw(const Tensor& x, int64_t H, int64_t W); // [N,H*W,C] -> [N,C,H,W]
Tensor slice_batch(const Tensor& x, int64_t b);            // x[b:b+1]
Tensor repeat_batch(const Tensor& x, int64_t times);       // [1,...] -> [times,...]

}  // namespace sd
