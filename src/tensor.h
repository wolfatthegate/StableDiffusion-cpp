// ============================================================================
// tensor.h - Minimal dense float32 tensor and every operator Stable Diffusion
// needs.
//
// Design notes
//   * A Tensor is just a shape plus a contiguous, row-major std::vector<float>.
//     There are no strides, views, dtypes or devices: every op allocates and
//     returns a fresh tensor (or works in place, for ops ending in '_').
//   * Two activation layouts are used throughout the model:
//       NCHW     [batch, channels, height, width]  for convolutions / norms
//       [N,L,C]  [batch, sequence length, features] for attention / linear
//     nchw_to_nlc() and nlc_to_nchw() convert between them.
//   * All heavy work funnels into gemm_nt(), so it is the single place to
//     optimise (or replace with a BLAS, see USE_ACCELERATE).
//   * Parallelism comes from one global thread pool (parallel_for).
// ============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sd {

// Dense, contiguous, row-major float32 tensor.
//
// The last dimension varies fastest: element (i0, i1, ..., ik) lives at
// data[((i0 * shape[1] + i1) * shape[2] + ...) + ik].
struct Tensor {
    std::vector<int64_t> shape;  // size of each dimension, outermost first
    std::vector<float> data;     // numel() elements, row-major

    Tensor() = default;
    // Allocates a zero-filled tensor of the given shape.
    explicit Tensor(std::vector<int64_t> s) : shape(std::move(s)), data(numel_of(shape), 0.0f) {}

    // Product of all dimensions (1 for an empty shape).
    static int64_t numel_of(const std::vector<int64_t>& s);
    int64_t numel() const { return (int64_t)data.size(); }
    int ndim() const { return (int)shape.size(); }
    // Size of dimension i; negative i counts from the end (dim(-1) = last).
    int64_t dim(int i) const { return shape[i < 0 ? i + ndim() : i]; }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    bool empty() const { return data.empty(); }
    // Changes the shape in place without moving data. Throws if the element
    // count differs.
    Tensor& reshape(std::vector<int64_t> s);
    // Human-readable shape such as "[1, 4, 64, 64]" for error messages.
    std::string shape_str() const;
};

// ---- threading -------------------------------------------------------------

// Sets the worker count for the global pool (0 = all hardware threads).
// The pool is rebuilt lazily on the next parallel call.
void set_num_threads(int n);
// Number of threads in the pool, including the calling thread.
int num_threads();
// Runs fn(i) for every i in [0, n). Work items are handed out dynamically
// through an atomic counter, and the calling thread participates. Blocks
// until all items are done. Not re-entrant: do not call from inside fn.
void parallel_for(int64_t n, const std::function<void(int64_t)>& fn);
// Splits [0, n) into contiguous ranges of at least min_chunk elements (and at
// most 4 ranges per thread) and runs fn(begin, end) on each range in parallel.
// Use this for cheap per-element loops where per-item dispatch would dominate.
void parallel_chunks(int64_t n, int64_t min_chunk, const std::function<void(int64_t, int64_t)>& fn);

// ---- GEMM ------------------------------------------------------------------

// General matrix multiply with the second operand transposed ("NT"):
//
//     C[M,N] = A[M,K] * B[N,K]^T  (+ bias[N] broadcast over rows)
//
// A, B and C are row-major with leading dimensions lda, ldb, ldc (the
// distance in floats between consecutive rows), which lets callers pass
// sub-matrices such as one attention head inside a wider tensor.
//
// If accumulate is true the product is added to the existing contents of C;
// otherwise C is overwritten. bias may be nullptr.
//
// "B transposed" is the natural layout for everything in this codebase:
//   * Linear layers:  y = x W^T, PyTorch stores W as [out, in].
//   * im2col conv:    out = W[Cout, K] * cols[pixels, K]^T.
//   * Attention:      scores = Q K^T, both [seq, head_dim].
void gemm_nt(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb,
             float* C, int64_t ldc, const float* bias = nullptr, bool accumulate = false);

// ---- layers ----------------------------------------------------------------

// Fully connected layer, PyTorch nn.Linear semantics.
//   x: [..., in]   W: [out, in]   b: [out] or nullptr   ->   [..., out]
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b);

// 2-D convolution, PyTorch nn.Conv2d semantics with square stride and padding.
//   x: [N, Cin, H, W]   W: [Cout, Cin, kh, kw]   b: [Cout] or nullptr
//   -> [N, Cout, (H + 2*pad - kh)/stride + 1, (W + 2*pad - kw)/stride + 1]
// Implemented as im2col followed by gemm_nt, processed in pixel chunks so the
// im2col buffer stays near 64 MB regardless of image size.
Tensor conv2d(const Tensor& x, const Tensor& W, const Tensor* b, int stride, int pad);

// Group normalisation (nn.GroupNorm). Channels are split into `groups` equal
// groups; each (sample, group) is normalised to zero mean and unit variance
// over its channels and spatial positions, then scaled per channel:
//   y = (x - mean) / sqrt(var + eps) * gamma[c] + beta[c]
//   x: [N, C, ...]   gamma, beta: [C]
Tensor group_norm(const Tensor& x, int groups, const Tensor& gamma, const Tensor& beta, float eps);

// Layer normalisation over the last dimension (nn.LayerNorm).
//   x: [..., C]   gamma, beta: [C]
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps);

// Multi-head scaled dot-product attention.
//   q: [N, Sq, inner]   k, v: [N, Skv, inner]   ->   [N, Sq, inner]
// inner is split into `heads` contiguous slices of size d = inner / heads,
// and for each head:  out_h = softmax(q_h k_h^T / sqrt(d)) v_h.
// With causal = true, query i may only attend to keys 0..i (used by CLIP).
// The projections (to_q etc.) are applied by the caller, not here.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, bool causal = false);

// GEGLU activation used in the UNet feed-forward layers.
// Splits the last dimension in half into (a, g) and returns a * gelu(g).
//   x: [..., 2D]   ->   [..., D]
Tensor geglu(const Tensor& x);

// ---- elementwise / shape ---------------------------------------------------

// In-place SiLU (swish): x * sigmoid(x).
void silu_(Tensor& x);
// In-place exact GELU: 0.5 x (1 + erf(x / sqrt 2)).
void gelu_(Tensor& x);
// In-place "quick GELU" used by OpenAI CLIP: x * sigmoid(1.702 x).
void quick_gelu_(Tensor& x);
// a += b elementwise. Shapes must have the same element count.
void add_(Tensor& a, const Tensor& b);
// x *= s elementwise.
void scale_(Tensor& x, float s);
// Numerically stable row-wise softmax over a [rows, cols] matrix, in place.
// Each logit is multiplied by `scale` first. With causal = true, row r keeps
// columns 0..r and sets the rest to 0.
void softmax_rows_(float* x, int64_t rows, int64_t cols, float scale, bool causal);
// Nearest-neighbour 2x upsampling: [N,C,H,W] -> [N,C,2H,2W].
Tensor upsample_nearest2x(const Tensor& x);
// Concatenates two NCHW tensors along the channel axis: a first, then b.
Tensor concat_channels(const Tensor& a, const Tensor& b);
// Concatenates along dimension 0 (batch): a first, then b.
Tensor concat_batch(const Tensor& a, const Tensor& b);
// [N,C,H,W] -> [N,H*W,C]: image to token sequence for attention.
Tensor nchw_to_nlc(const Tensor& x);
// [N,H*W,C] -> [N,C,H,W]: inverse of nchw_to_nlc.
Tensor nlc_to_nchw(const Tensor& x, int64_t H, int64_t W);
// Copies batch element b out as a tensor with leading dimension 1.
Tensor slice_batch(const Tensor& x, int64_t b);
// Tiles the whole tensor `times` times along dimension 0.
Tensor repeat_batch(const Tensor& x, int64_t times);

}  // namespace sd
