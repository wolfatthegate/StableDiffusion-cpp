// ============================================================================
// backend_cuda.cu - CUDA implementation of backend_cuda.h.
//
// Design: plain kernels for everything except matrix products, which go to
// cuBLAS (float32 SGEMM, or bf16 tensor-core GemmEx with float32
// accumulation for bf16 weights). Attention batches all heads into strided
// batched GEMMs (one launch for the scores, one softmax, one for the weighted
// sum), in chunks that bound the score buffer. Reductions use float32 warp
// shuffles. A size-class caching allocator keeps cudaMalloc, which
// synchronises the device, off the hot path. Everything runs on the default
// stream, in order; device-to-device copies and memsets are asynchronous.
// Env:   SD_CUDA_ATTN_MB=<n> caps the attention score buffer (default 1024).
//
// Build: make CUDA=1   (nvcc -O3 -std=c++17 -arch=native; links cudart, cublas)
// Test:  ./test_ops    (compares every op with the CPU implementation)
// Env:   SD_CUDA_TF32=1 allows TF32 tensor cores for float32 GEMMs (faster,
//        about 1e-3 relative error).
// ============================================================================
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend_cuda.h"

namespace sd {
namespace cuda {

// ---------------------------------------------------------------------------
// Error handling, device state, allocator
// ---------------------------------------------------------------------------
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS error in ") + what + ": status " + std::to_string((int)s));
}
void launched(const char* what) { check(cudaGetLastError(), what); }

struct State {
    bool probed = false, ok = false;
    cublasHandle_t blas = nullptr;
    std::string name;
    size_t memory = 0;
    std::mutex mutex;
    std::map<void*, size_t> live;                      // allocation -> size
    std::map<size_t, std::vector<void*>> free_lists;   // size -> idle blocks
    size_t cached_bytes = 0;
    static constexpr size_t kMaxCached = (size_t)6 << 30;  // idle blocks kept, 6 GB

    void probe() {
        if (probed) return;
        probed = true;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
            cudaGetLastError();
            return;
        }
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return;
        name = prop.name;
        memory = prop.totalGlobalMem;
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return;
        if (std::getenv("SD_CUDA_TF32")) cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);
        ok = true;
    }
};

State& st() {
    static State s;
    s.probe();
    return s;
}

cublasHandle_t blas() {
    State& s = st();
    if (!s.ok) throw std::runtime_error("CUDA device not available");
    return s.blas;
}

constexpr int kThreads = 256;
inline unsigned grid_for(int64_t n) { return (unsigned)std::min<int64_t>((n + kThreads - 1) / kThreads, 65535 * 16); }

// Scratch buffer released at the end of the scope.
struct Scratch {
    void* p = nullptr;
    explicit Scratch(size_t bytes) { p = cuda::malloc(bytes); }
    ~Scratch() { cuda::free(p); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    float* f() { return (float*)p; }
    uint16_t* h() { return (uint16_t*)p; }
};

}  // namespace

bool available() { return st().ok; }
const char* device_name() { return st().name.c_str(); }
size_t device_memory() { return st().memory; }

namespace {
// Size classes: round up to the next 1/8 of a power of two (at most 12.5%
// waste), so blocks freed by one call are reused by the next similar one.
size_t size_class(size_t bytes) {
    bytes = std::max<size_t>(bytes, 256);
    size_t p = 256;
    while (p < bytes) p <<= 1;
    const size_t step = std::max<size_t>(256, p / 8);
    return (bytes + step - 1) / step * step;
}
}  // namespace

void* malloc(size_t bytes) {
    State& s = st();
    if (!s.ok) throw std::runtime_error("CUDA device not available");
    bytes = size_class(bytes);
    {
        std::lock_guard<std::mutex> lk(s.mutex);
        auto it = s.free_lists.find(bytes);
        if (it != s.free_lists.end() && !it->second.empty()) {
            void* p = it->second.back();
            it->second.pop_back();
            s.cached_bytes -= bytes;
            s.live[p] = bytes;
            return p;
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {  // drop the cache and retry once
        cudaGetLastError();
        std::lock_guard<std::mutex> lk(s.mutex);
        for (auto& kv : s.free_lists)
            for (void* q : kv.second) cudaFree(q);
        s.free_lists.clear();
        s.cached_bytes = 0;
        check(cudaMalloc(&p, bytes), "cudaMalloc");
    }
    std::lock_guard<std::mutex> lk(s.mutex);
    s.live[p] = bytes;
    return p;
}

void free(void* p) {
    if (!p) return;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mutex);
    auto it = s.live.find(p);
    const size_t bytes = it == s.live.end() ? 0 : it->second;
    if (it != s.live.end()) s.live.erase(it);
    if (bytes && s.cached_bytes + bytes <= State::kMaxCached) {
        s.free_lists[bytes].push_back(p);
        s.cached_bytes += bytes;
    } else {
        cudaFree(p);
    }
}

void upload(void* dst, const void* src, size_t bytes) { check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "upload"); }
void download(void* dst, const void* src, size_t bytes) { check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "download"); }
// Device-side copies and fills are queued on the stream without waiting.
void copy(void* dst, const void* src, size_t bytes) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0), "copy");
}
void zero(void* dst, size_t bytes) { check(cudaMemsetAsync(dst, 0, bytes, 0), "memset"); }
void synchronize() { check(cudaDeviceSynchronize(), "synchronize"); }

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
namespace {

__device__ inline float bf16_to_f32_d(uint16_t h) { return __uint_as_float((unsigned)h << 16); }
__device__ inline uint16_t f32_to_bf16_d(float f) {
    unsigned bits = __float_as_uint(f);
    if ((bits & 0x7F800000u) == 0x7F800000u) return (uint16_t)(bits >> 16);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

// Block-wide reductions in float32 with warp shuffles: one shuffle tree per
// warp, then one over the warp results. blockDim.x must be a multiple of 32
// (every launch here uses 32 to 1024 threads). Float32 is accurate enough:
// each thread accumulates a strided partial sum first, so the tree adds at
// most a few thousand partials, and double precision runs at 1/32 to 1/64 of
// float speed on consumer GPUs.
__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ inline float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v) {
    __shared__ float sh[32];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[lane] : -3.0e38f;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

#define GRID_STRIDE(i, n) for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (n); i += (int64_t)gridDim.x * blockDim.x)

// ---- GEMM helpers ----------------------------------------------------------
__global__ void k_pack_bf16(const float* A, int64_t lda, uint16_t* out, int64_t M, int64_t K) {
    GRID_STRIDE(idx, M * K) { out[idx] = f32_to_bf16_d(A[(idx / K) * lda + idx % K]); }
}
__global__ void k_add_bias_cols(float* C, int64_t ldc, int64_t M, int64_t N, const float* bias) {
    GRID_STRIDE(idx, M * N) { C[(idx / N) * ldc + idx % N] += bias[idx % N]; }
}
__global__ void k_add_bias_rows(float* C, int64_t ldc, int64_t M, int64_t N, const float* bias) {
    GRID_STRIDE(idx, M * N) { C[(idx / N) * ldc + idx % N] += bias[idx / N]; }
}

// Row-major C[M,N] = A[M,K] * B[N,K]^T, through column-major cuBLAS: C^T = B^T-view * A-view.
void sgemm_nt(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
              int64_t ldc, float beta) {
    const float alpha = 1.0f;
    check(cublasSgemm(blas(), CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, B, (int)ldb, A, (int)lda, &beta, C,
                      (int)ldc),
          "cublasSgemm NT");
}
// Row-major C[M,N] = A[M,K] * B[K,N].
void sgemm_nn(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
              int64_t ldc, float beta) {
    const float alpha = 1.0f;
    check(cublasSgemm(blas(), CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, B, (int)ldb, A, (int)lda, &beta, C,
                      (int)ldc),
          "cublasSgemm NN");
}

// ---- im2col ------------------------------------------------------------------
__global__ void k_im2col(const float* x, float* cols, int64_t Cin, int64_t H, int64_t W, int64_t kh, int64_t kw, int64_t Wo,
                         int stride, int pad, int64_t p0, int64_t P, int64_t K) {
    GRID_STRIDE(idx, P * K) {
        const int64_t p = idx / K, kk = idx % K;
        const int64_t ci = kk / (kh * kw), r = kk % (kh * kw), ky = r / kw, kx = r % kw;
        const int64_t op = p0 + p, oy = op / Wo, ox = op % Wo;
        const int64_t iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
        cols[idx] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[(ci * H + iy) * W + ix] : 0.0f;
    }
}

// ---- norms -------------------------------------------------------------------
__global__ void k_group_norm(const float* x, float* y, int64_t C, int64_t HW, int groups, const float* gamma,
                             const float* beta, float eps) {
    const int64_t n = blockIdx.x / groups, g = blockIdx.x % groups, cg = C / groups, len = cg * HW;
    const float* p = x + (n * C + g * cg) * HW;
    float* q = y + (n * C + g * cg) * HW;
    float s = 0;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) s += p[i];
    const float mean = block_sum(s) / (float)len;
    float v = 0;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        const float d = p[i] - mean;
        v += d * d;
    }
    const float rstd = rsqrtf(block_sum(v) / (float)len + eps);
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        const int64_t c = g * cg + i / HW;
        q[i] = (p[i] - mean) * rstd * gamma[c] + beta[c];
    }
}

// Layer norm (rms = false) or RMS norm (rms = true) of one row per block.
__global__ void k_row_norm(const float* x, float* y, int64_t C, const float* gamma, const float* beta, float eps, bool rms) {
    const float* p = x + (int64_t)blockIdx.x * C;
    float* q = y + (int64_t)blockIdx.x * C;
    float mean = 0;
    if (!rms) {
        float s = 0;
        for (int64_t i = threadIdx.x; i < C; i += blockDim.x) s += p[i];
        mean = block_sum(s) / (float)C;
    }
    float v = 0;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        const float d = p[i] - mean;
        v += d * d;
    }
    const float rstd = rsqrtf(block_sum(v) / (float)C + eps);
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float o = (p[i] - mean) * rstd;
        if (gamma) o *= gamma[i];
        if (beta) o += beta[i];
        q[i] = o;
    }
}

__global__ void k_rms_norm_heads(float* x, int64_t dim, const float* gamma, float eps) {
    float* p = x + (int64_t)blockIdx.x * dim;
    float ss = 0;
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) ss += p[i] * p[i];
    const float rstd = rsqrtf(block_sum(ss) / (float)dim + eps);
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) p[i] = p[i] * rstd * (gamma ? gamma[i] : 1.0f);
}

// ---- rotary, softmax, activations -----------------------------------------------
__global__ void k_rope(float* x, int64_t NS, int64_t S, int64_t inner, int64_t dim, const float* cs, const float* sn,
                       bool interleaved) {
    const int64_t heads = inner / dim, half = dim / 2;
    GRID_STRIDE(idx, NS * heads * half) {
        const int64_t i = idx % half, h = (idx / half) % heads, ns = idx / (half * heads), s = ns % S;
        float* p = x + ns * inner + h * dim;
        const float* c = cs + s * dim;
        const float* sv = sn + s * dim;
        if (interleaved) {
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i] = a * c[2 * i] - b * sv[2 * i];
            p[2 * i + 1] = b * c[2 * i] + a * sv[2 * i];
        } else {
            const float a = p[i], b = p[i + half];
            p[i] = a * c[i] - b * sv[i];
            p[i + half] = b * c[i + half] + a * sv[i + half];
        }
    }
}

// One block per row. `period` is the number of query rows per matrix, so a
// stack of per-head score matrices gets the causal mask row by row.
__global__ void k_softmax_rows(float* x, int64_t cols, float scale, bool causal, int64_t period) {
    const int64_t r = blockIdx.x;
    float* p = x + r * cols;
    const int64_t valid = causal ? min(cols, (r % period) + 1) : cols;
    float m = -3.0e38f;
    for (int64_t j = threadIdx.x; j < valid; j += blockDim.x) m = fmaxf(m, p[j] * scale);
    m = block_max(m);
    float s = 0;
    for (int64_t j = threadIdx.x; j < valid; j += blockDim.x) {
        const float e = expf(p[j] * scale - m);
        p[j] = e;
        s += e;
    }
    const float inv = 1.0f / block_sum(s);
    for (int64_t j = threadIdx.x; j < valid; j += blockDim.x) p[j] *= inv;
    for (int64_t j = valid + threadIdx.x; j < cols; j += blockDim.x) p[j] = 0.0f;
}

__global__ void k_geglu(const float* x, float* y, int64_t rows, int64_t D) {
    GRID_STRIDE(idx, rows * D) {
        const int64_t r = idx / D, i = idx % D;
        const float a = x[r * 2 * D + i], g = x[r * 2 * D + D + i];
        y[idx] = a * 0.5f * g * (1.0f + erff(g * 0.70710678118654752f));
    }
}
__global__ void k_swiglu(const float* g, const float* u, float* y, int64_t n) {
    GRID_STRIDE(i, n) { y[i] = g[i] / (1.0f + expf(-g[i])) * u[i]; }
}
__global__ void k_unary(float* x, int64_t n, int kind) {
    GRID_STRIDE(i, n) {
        const float v = x[i];
        if (kind == 0) x[i] = v / (1.0f + expf(-v));
        else if (kind == 1) x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        else if (kind == 2) x[i] = v / (1.0f + expf(-1.702f * v));
        else x[i] = tanhf(v);
    }
}
__global__ void k_add(float* a, const float* b, int64_t n) {
    GRID_STRIDE(i, n) { a[i] += b[i]; }
}
__global__ void k_affine(float* x, float a, float b, int64_t n) {
    GRID_STRIDE(i, n) { x[i] = x[i] * a + b; }
}
__global__ void k_mul_rows(float* x, const float* v, int64_t n, int64_t dim) {
    GRID_STRIDE(i, n) { x[i] *= v[i % dim]; }
}
__global__ void k_add_rows_scaled(float* x, const float* y, const float* g, int64_t n, int64_t dim) {
    GRID_STRIDE(i, n) { x[i] += g[i % dim] * y[i]; }
}
__global__ void k_add_channel_bias(float* x, const float* b, int64_t n, int64_t HW) {
    GRID_STRIDE(i, n) { x[i] += b[i / HW]; }
}

// ---- shape ops ---------------------------------------------------------------------
__global__ void k_upsample2x(const float* x, float* y, int64_t NC, int64_t H, int64_t W) {
    const int64_t H2 = 2 * H, W2 = 2 * W;
    GRID_STRIDE(idx, NC * H2 * W2) {
        const int64_t nc = idx / (H2 * W2), r = idx % (H2 * W2), i = r / W2, j = r % W2;
        y[idx] = x[(nc * H + i / 2) * W + j / 2];
    }
}
__global__ void k_concat_channels(const float* a, const float* b, float* y, int64_t N, int64_t Ca, int64_t Cb, int64_t HW) {
    const int64_t C = Ca + Cb;
    GRID_STRIDE(idx, N * C * HW) {
        const int64_t n = idx / (C * HW), c = (idx / HW) % C, i = idx % HW;
        y[idx] = c < Ca ? a[(n * Ca + c) * HW + i] : b[(n * Cb + (c - Ca)) * HW + i];
    }
}
__global__ void k_nchw_to_nlc(const float* x, float* y, int64_t N, int64_t C, int64_t HW) {
    GRID_STRIDE(idx, N * C * HW) {  // idx indexes the output [N, HW, C]
        const int64_t n = idx / (HW * C), l = (idx / C) % HW, c = idx % C;
        y[idx] = x[(n * C + c) * HW + l];
    }
}
__global__ void k_nlc_to_nchw(const float* x, float* y, int64_t N, int64_t C, int64_t HW) {
    GRID_STRIDE(idx, N * C * HW) {  // idx indexes the output [N, C, HW]
        const int64_t n = idx / (C * HW), c = (idx / HW) % C, l = idx % HW;
        y[idx] = x[(n * HW + l) * C + c];
    }
}
__global__ void k_patchify(const float* x, float* y, int64_t C, int64_t H, int64_t W, int p) {
    const int64_t Wt = W / p, F = (int64_t)p * p * C;
    GRID_STRIDE(idx, (H / p) * Wt * F) {  // idx indexes tokens [n, F]
        const int64_t t = idx / F, f = idx % F, h = t / Wt, w = t % Wt;
        const int64_t c = f % C, pw = (f / C) % p, ph = f / (C * p);
        y[idx] = x[(c * H + h * p + ph) * W + w * p + pw];
    }
}
__global__ void k_unpatchify(const float* x, float* y, int64_t C, int64_t H, int64_t W, int p) {
    const int64_t Wt = W / p, F = (int64_t)p * p * C;
    GRID_STRIDE(idx, C * H * W) {  // idx indexes the image [C, H, W]
        const int64_t c = idx / (H * W), hh = (idx / W) % H, ww = idx % W;
        const int64_t h = hh / p, ph = hh % p, w = ww / p, pw = ww % p;
        y[idx] = x[(h * Wt + w) * F + (ph * p + pw) * C + c];
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public ops
// ---------------------------------------------------------------------------
void gemm_nt_f32(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
                 int64_t ldc, const float* bias, bool accumulate) {
    if (M <= 0 || N <= 0) return;
    sgemm_nt(M, N, K, A, lda, B, ldb, C, ldc, accumulate ? 1.0f : 0.0f);
    if (bias) {
        k_add_bias_cols<<<grid_for(M * N), kThreads>>>(C, ldc, M, N, bias);
        launched("add_bias_cols");
    }
}

void gemm_nt_bf16(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const uint16_t* B, int64_t ldb,
                  float* C, int64_t ldc, const float* bias, bool accumulate) {
    if (M <= 0 || N <= 0) return;
    Scratch Abf((size_t)(M * K) * sizeof(uint16_t));
    k_pack_bf16<<<grid_for(M * K), kThreads>>>(A, lda, Abf.h(), M, K);
    launched("pack_bf16");
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    check(cublasGemmEx(blas(), CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, B, CUDA_R_16BF, (int)ldb, Abf.h(),
                       CUDA_R_16BF, (int)K, &beta, C, CUDA_R_32F, (int)ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "cublasGemmEx bf16");
    if (bias) {
        k_add_bias_cols<<<grid_for(M * N), kThreads>>>(C, ldc, M, N, bias);
        launched("add_bias_cols");
    }
}

void conv2d(const float* x, int64_t N, int64_t Cin, int64_t H, int64_t W, const float* w, int64_t Cout, int64_t kh,
            int64_t kw, const float* bias, int stride, int pad, float* out, int64_t Ho, int64_t Wo) {
    const int64_t K = Cin * kh * kw, HW = Ho * Wo;
    if (kh == 1 && kw == 1 && stride == 1 && pad == 0) {
        // A 1x1 convolution is out[Cout, HW] = W[Cout, Cin] * x[Cin, HW]: no im2col.
        for (int64_t n = 0; n < N; ++n) {
            float* o = out + n * Cout * HW;
            sgemm_nn(Cout, HW, Cin, w, Cin, x + n * Cin * H * W, HW, o, HW, 0.0f);
            if (bias) {
                k_add_bias_rows<<<grid_for(Cout * HW), kThreads>>>(o, HW, Cout, HW, bias);
                launched("add_bias_rows");
            }
        }
        return;
    }
    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(HW, (int64_t(1) << 26) / K));  // <= 256 MB of columns
    Scratch cols((size_t)(chunk * K) * sizeof(float));
    for (int64_t n = 0; n < N; ++n) {
        const float* xin = x + n * Cin * H * W;
        float* o = out + n * Cout * HW;
        for (int64_t p0 = 0; p0 < HW; p0 += chunk) {
            const int64_t P = std::min(chunk, HW - p0);
            k_im2col<<<grid_for(P * K), kThreads>>>(xin, cols.f(), Cin, H, W, kh, kw, Wo, stride, pad, p0, P, K);
            launched("im2col");
            sgemm_nt(Cout, P, K, w, K, cols.f(), K, o + p0, HW, 0.0f);
        }
        if (bias) {
            k_add_bias_rows<<<grid_for(Cout * HW), kThreads>>>(o, HW, Cout, HW, bias);
            launched("add_bias_rows");
        }
    }
}

void group_norm(const float* x, float* y, int64_t N, int64_t C, int64_t HW, int groups, const float* gamma,
                const float* beta, float eps) {
    k_group_norm<<<(unsigned)(N * groups), 512>>>(x, y, C, HW, groups, gamma, beta, eps);
    launched("group_norm");
}

void layer_norm(const float* x, float* y, int64_t rows, int64_t C, const float* gamma, const float* beta, float eps) {
    if (rows <= 0) return;
    k_row_norm<<<(unsigned)rows, kThreads>>>(x, y, C, gamma, beta, eps, false);
    launched("layer_norm");
}

void rms_norm(const float* x, float* y, int64_t rows, int64_t C, const float* gamma, float eps) {
    if (rows <= 0) return;
    k_row_norm<<<(unsigned)rows, kThreads>>>(x, y, C, gamma, nullptr, eps, true);
    launched("rms_norm");
}

void rms_norm_heads(float* x, int64_t groups, int64_t dim, const float* gamma, float eps) {
    if (groups <= 0) return;
    unsigned threads = 32;  // block_sum needs a power of two
    while (threads < dim && threads < 256) threads *= 2;
    k_rms_norm_heads<<<(unsigned)groups, threads>>>(x, dim, gamma, eps);
    launched("rms_norm_heads");
}

void rope(float* x, int64_t N, int64_t S, int64_t inner, int64_t dim, const float* cs, const float* sn, bool interleaved) {
    const int64_t total = N * S * (inner / dim) * (dim / 2);
    k_rope<<<grid_for(total), kThreads>>>(x, N * S, S, inner, dim, cs, sn, interleaved);
    launched("rope");
}

void softmax_rows(float* x, int64_t rows, int64_t cols, float scale, bool causal) {
    if (rows <= 0) return;
    k_softmax_rows<<<(unsigned)rows, kThreads>>>(x, cols, scale, causal, rows);
    launched("softmax_rows");
}

void attention(const float* q, const float* k, const float* v, float* out, int64_t N, int64_t Sq, int64_t Skv, int heads,
               int kv_heads, int64_t d, bool causal) {
    const int64_t inner = heads * d, kv_inner = (int64_t)kv_heads * d, group = heads / kv_heads;
    const float scale = 1.0f / sqrtf((float)d), one = 1.0f, zero_f = 0.0f;
    // Query head h = kvh * group + j uses key/value head kvh. For a fixed j the
    // heads kvh = 0..kv_heads-1 are evenly strided in q, k, v and out, so they
    // form one strided batch: q and out step by group*d, k and v by d.
    // Batches are chunked so the score buffer stays under SD_CUDA_ATTN_MB.
    static const size_t budget = [] {
        const char* e = std::getenv("SD_CUDA_ATTN_MB");
        return (size_t)(e ? std::max(64, std::atoi(e)) : 1024) << 20;
    }();
    const size_t per_head = (size_t)(Sq * Skv) * sizeof(float);
    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(kv_heads, (int64_t)(budget / per_head)));
    Scratch scores(per_head * (size_t)chunk);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t j = 0; j < group; ++j)
            for (int64_t h0 = 0; h0 < kv_heads; h0 += chunk) {
                const int64_t B = std::min<int64_t>(chunk, kv_heads - h0);
                const float* qb = q + n * Sq * inner + (h0 * group + j) * d;
                const float* kb = k + n * Skv * kv_inner + h0 * d;
                const float* vb = v + n * Skv * kv_inner + h0 * d;
                float* ob = out + n * Sq * inner + (h0 * group + j) * d;
                // scores_b[Sq, Skv] = Q_b K_b^T  (column-major view: S^T = K Q^T)
                check(cublasSgemmStridedBatched(blas(), CUBLAS_OP_T, CUBLAS_OP_N, (int)Skv, (int)Sq, (int)d, &one, kb,
                                                (int)kv_inner, d, qb, (int)inner, group * d, &zero_f, scores.f(), (int)Skv,
                                                Sq * Skv, (int)B),
                      "attention scores");
                k_softmax_rows<<<(unsigned)(B * Sq), kThreads>>>(scores.f(), Skv, scale, causal, Sq);
                launched("attention softmax");
                // out_b[Sq, d] = P_b V_b  (column-major view: O^T = V^T P^T)
                check(cublasSgemmStridedBatched(blas(), CUBLAS_OP_N, CUBLAS_OP_N, (int)d, (int)Sq, (int)Skv, &one, vb,
                                                (int)kv_inner, d, scores.f(), (int)Skv, Sq * Skv, &zero_f, ob, (int)inner,
                                                group * d, (int)B),
                      "attention values");
            }
}

void geglu(const float* x, float* y, int64_t rows, int64_t D) {
    k_geglu<<<grid_for(rows * D), kThreads>>>(x, y, rows, D);
    launched("geglu");
}
void swiglu(const float* g, const float* u, float* y, int64_t n) {
    k_swiglu<<<grid_for(n), kThreads>>>(g, u, y, n);
    launched("swiglu");
}
void silu(float* x, int64_t n) { k_unary<<<grid_for(n), kThreads>>>(x, n, 0); launched("silu"); }
void tanh(float* x, int64_t n) { k_unary<<<grid_for(n), kThreads>>>(x, n, 3); launched("tanh"); }
void gelu(float* x, int64_t n) { k_unary<<<grid_for(n), kThreads>>>(x, n, 1); launched("gelu"); }
void quick_gelu(float* x, int64_t n) { k_unary<<<grid_for(n), kThreads>>>(x, n, 2); launched("quick_gelu"); }
void add(float* a, const float* b, int64_t n) { k_add<<<grid_for(n), kThreads>>>(a, b, n); launched("add"); }
void scale(float* x, float s, int64_t n) { k_affine<<<grid_for(n), kThreads>>>(x, s, 0.0f, n); launched("scale"); }
void affine(float* x, float a, float b, int64_t n) { k_affine<<<grid_for(n), kThreads>>>(x, a, b, n); launched("affine"); }
void mul_rows(float* x, const float* v, int64_t rows, int64_t dim) {
    k_mul_rows<<<grid_for(rows * dim), kThreads>>>(x, v, rows * dim, dim);
    launched("mul_rows");
}
void add_rows_scaled(float* x, const float* y, const float* g, int64_t rows, int64_t dim) {
    k_add_rows_scaled<<<grid_for(rows * dim), kThreads>>>(x, y, g, rows * dim, dim);
    launched("add_rows_scaled");
}
void add_channel_bias(float* x, const float* b, int64_t N, int64_t C, int64_t HW) {
    k_add_channel_bias<<<grid_for(N * C * HW), kThreads>>>(x, b, N * C * HW, HW);
    launched("add_channel_bias");
}
void upsample_nearest2x(const float* x, float* y, int64_t NC, int64_t H, int64_t W) {
    k_upsample2x<<<grid_for(NC * 4 * H * W), kThreads>>>(x, y, NC, H, W);
    launched("upsample_nearest2x");
}
void concat_channels(const float* a, const float* b, float* y, int64_t N, int64_t Ca, int64_t Cb, int64_t HW) {
    k_concat_channels<<<grid_for(N * (Ca + Cb) * HW), kThreads>>>(a, b, y, N, Ca, Cb, HW);
    launched("concat_channels");
}
void nchw_to_nlc(const float* x, float* y, int64_t N, int64_t C, int64_t HW) {
    k_nchw_to_nlc<<<grid_for(N * C * HW), kThreads>>>(x, y, N, C, HW);
    launched("nchw_to_nlc");
}
void nlc_to_nchw(const float* x, float* y, int64_t N, int64_t C, int64_t HW) {
    k_nlc_to_nchw<<<grid_for(N * C * HW), kThreads>>>(x, y, N, C, HW);
    launched("nlc_to_nchw");
}
void patchify(const float* x, float* y, int64_t C, int64_t H, int64_t W, int p) {
    k_patchify<<<grid_for(C * H * W), kThreads>>>(x, y, C, H, W, p);
    launched("patchify");
}
void unpatchify(const float* x, float* y, int64_t C, int64_t H, int64_t W, int p) {
    k_unpatchify<<<grid_for(C * H * W), kThreads>>>(x, y, C, H, W, p);
    launched("unpatchify");
}

}  // namespace cuda
}  // namespace sd
