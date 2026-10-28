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
//   * A Tensor may live on the GPU (Device::CUDA). Every op then runs the
//     CUDA implementation from backend_cuda.h and returns a GPU tensor; ops
//     never mix devices. to(Device) copies across. The CPU path is unchanged
//     and is the only one on machines without CUDA.
// ============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sd {

// Where a tensor's memory lives.
enum class Device { CPU, CUDA };
const char* device_name(Device d);          // "cpu" / "cuda"
Device parse_device(const std::string& s);  // "cpu", "cuda", or "auto" (cuda if available)

// Owns one GPU allocation (defined in tensor.cpp).
struct DeviceBuffer;

// Dense, contiguous, row-major float32 tensor.
//
// The last dimension varies fastest: element (i0, i1, ..., ik) lives at
// data[((i0 * shape[1] + i1) * shape[2] + ...) + ik].
// Element type. Activations are always F32. Large model weights may be kept
// as BF16 (the top 16 bits of a float32) to halve memory; only the ops that
// document it (linear, embedding lookup) accept BF16 inputs.
enum class DType { F32, BF16 };

struct Tensor {
    std::vector<int64_t> shape;   // size of each dimension, outermost first
    std::vector<float> data;      // F32 host storage: numel() elements, row-major
    std::vector<uint16_t> data16; // BF16 host storage, used when dtype == BF16
    DType dtype = DType::F32;
    Device device = Device::CPU;
    std::shared_ptr<DeviceBuffer> dev;  // GPU storage when device == CUDA (host vectors are then empty)

    Tensor() = default;
    // Allocates a zero-filled F32 tensor of the given shape on the CPU.
    explicit Tensor(std::vector<int64_t> s) : shape(std::move(s)), data(numel_of(shape), 0.0f) {}
    // Allocates a zero-filled F32 tensor on the given device.
    Tensor(std::vector<int64_t> s, Device d);

    // Product of all dimensions (1 for an empty shape).
    static int64_t numel_of(const std::vector<int64_t>& s);
    int64_t numel() const {
        if (device == Device::CUDA) return numel_of(shape);
        return dtype == DType::F32 ? (int64_t)data.size() : (int64_t)data16.size();
    }
    bool is_bf16() const { return dtype == DType::BF16; }
    bool is_cuda() const { return device == Device::CUDA; }
    // Copies the tensor to a device (a plain copy when already there).
    Tensor to(Device d) const;
    // Device memory pointers (CUDA tensors only).
    float* dptr();
    const float* dptr() const;
    const uint16_t* dptr16() const;
    // Converts BF16 storage to F32 in place (no-op for F32).
    void to_f32();
    // Converts F32 storage to BF16 in place (rounds to nearest even).
    void to_bf16();
    // Reads element i as float whatever the storage type.
    float at(int64_t i) const;
    int ndim() const { return (int)shape.size(); }
    // Size of dimension i; negative i counts from the end (dim(-1) = last).
    int64_t dim(int i) const { return shape[i < 0 ? i + ndim() : i]; }
    float* ptr() { check_host(); return data.data(); }              // F32, CPU only
    const float* ptr() const { check_host(); return data.data(); }  // F32, CPU only
    const uint16_t* ptr16() const { check_host(); return data16.data(); }
    bool empty() const { return numel() == 0; }
    void check_host() const;  // throws if the tensor is on the GPU
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
// Same with a BF16 B operand (model weights). Tiles of B are expanded to
// float32 in a per-thread buffer and multiplied with the float32 kernel.
void gemm_nt_bf16(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const uint16_t* B, int64_t ldb,
                  float* C, int64_t ldc, const float* bias = nullptr, bool accumulate = false);
// bf16 <-> f32 scalar conversions.
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
uint16_t f32_to_bf16(float f);

// ---- layers ----------------------------------------------------------------

// Fully connected layer, PyTorch nn.Linear semantics. W may be F32 or BF16.
//   x: [..., in]   W: [out, in]   b: [out] or nullptr   ->   [..., out]
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b);

// Row lookup: out[i] = table[ids[i]] for an F32 or BF16 table [vocab, D].
Tensor embedding(const Tensor& table, const std::vector<int32_t>& ids);

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
//   x: [..., C]   gamma, beta: [C]; either may be nullptr for no affine
Tensor layer_norm(const Tensor& x, const Tensor* gamma, const Tensor* beta, float eps);
inline Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    return layer_norm(x, &gamma, &beta, eps);
}

// RMS normalisation over the last dimension: x / sqrt(mean(x^2) + eps) * gamma.
//   x: [..., C]   gamma: [C] or nullptr
Tensor rms_norm(const Tensor& x, const Tensor* gamma, float eps);
// RMS normalisation over the last `dim` elements of each row of `width`
// (per-head QK-norm: x [..., heads*dim], gamma [dim]).
void rms_norm_heads_(Tensor& x, int64_t dim, const Tensor* gamma, float eps);

// Rotary position embedding applied in place to x [S, heads*dim] (or
// [N, S, heads*dim]) using cos/sin tables [S, dim].
//   interleaved = false: rotate_half convention (pairs i and i + dim/2), as
//                        in Llama / Qwen.
//   interleaved = true:  complex-pair convention (pairs 2i and 2i+1), as in
//                        FLUX / Z-Image. Tables then hold each angle twice.
void rope_(Tensor& x, int64_t dim, const Tensor& cos, const Tensor& sin, bool interleaved);

// SwiGLU: silu(gate) * up, elementwise, same shapes.
Tensor swiglu(const Tensor& gate, const Tensor& up);

// Row-vector helpers, all in place. Rows are the leading dimensions of x
// flattened; dim is x's last dimension.
void affine_(Tensor& x, float a, float b);                         // x = x * a + b
void mul_rows_(Tensor& x, const Tensor& v);                        // x[r, :] *= v  (v has dim elements)
void add_rows_scaled_(Tensor& x, const Tensor& y, const Tensor& g); // x[r, :] += g * y[r, :]
void add_channel_bias_(Tensor& x, const Tensor& b);                // x [N,C,H,W] += b [N,C] per channel
// [1, C, H, W] -> [(H/p)*(W/p), p*p*C] with features ordered (ph, pw, c), and back.
Tensor patchify2d(const Tensor& x, int p);
Tensor unpatchify2d(const Tensor& tokens, int64_t C, int64_t H, int64_t W, int p);
// Rows [from, to) of x viewed as [rows, last dim] -> [to - from, last dim].
Tensor slice_rows(const Tensor& x, int64_t from, int64_t to);

// Multi-head scaled dot-product attention.
//   q: [N, Sq, heads*d]   k, v: [N, Skv, kv_heads*d]   ->   [N, Sq, heads*d]
// For each head:  out_h = softmax(q_h k_h^T / sqrt(d)) v_h. With kv_heads <
// heads (grouped-query attention) consecutive query heads share one key/value
// head; kv_heads = 0 means kv_heads = heads.
// With causal = true, query i may only attend to keys 0..i.
// The projections (to_q etc.) are applied by the caller, not here.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, bool causal = false, int kv_heads = 0);

// GEGLU activation used in the UNet feed-forward layers.
// Splits the last dimension in half into (a, g) and returns a * gelu(g).
//   x: [..., 2D]   ->   [..., D]
Tensor geglu(const Tensor& x);

// ---- elementwise / shape ---------------------------------------------------

// In-place SiLU (swish): x * sigmoid(x).
void silu_(Tensor& x);
// In-place hyperbolic tangent.
void tanh_(Tensor& x);
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
