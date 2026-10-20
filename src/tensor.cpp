#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef USE_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace sd {

int64_t Tensor::numel_of(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    return n;
}

Tensor& Tensor::reshape(std::vector<int64_t> s) {
    if (numel_of(s) != numel()) throw std::runtime_error("reshape: element count mismatch for " + shape_str());
    shape = std::move(s);
    return *this;
}

std::string Tensor::shape_str() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
    os << "]";
    return os.str();
}

// ============================================================================
// Thread pool
// ============================================================================
namespace {

class ThreadPool {
public:
    explicit ThreadPool(int n) : nthreads_(std::max(1, n)) {
        for (int i = 1; i < nthreads_; ++i) workers_.emplace_back([this] { worker(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int size() const { return nthreads_; }

    void run(int64_t n, const std::function<void(int64_t)>& fn) {
        if (n <= 0) return;
        if (n == 1 || workers_.empty()) {
            for (int64_t i = 0; i < n; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            n_ = n;
            next_.store(0, std::memory_order_relaxed);
            remaining_ = (int)workers_.size();
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return remaining_ == 0; });
        fn_ = nullptr;
    }

private:
    void work() {
        int64_t i;
        while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < n_) (*fn_)(i);
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            lk.unlock();
            work();
            lk.lock();
            if (--remaining_ == 0) done_cv_.notify_one();
        }
    }

    int nthreads_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int64_t)>* fn_ = nullptr;
    int64_t n_ = 0;
    std::atomic<int64_t> next_{0};
    int remaining_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

std::unique_ptr<ThreadPool> g_pool;
int g_requested_threads = 0;

ThreadPool& pool() {
    if (!g_pool) {
        int n = g_requested_threads > 0 ? g_requested_threads : (int)std::thread::hardware_concurrency();
        g_pool.reset(new ThreadPool(n < 1 ? 1 : n));
    }
    return *g_pool;
}

}  // namespace

void set_num_threads(int n) {
    g_requested_threads = n;
    g_pool.reset();
}
int num_threads() { return pool().size(); }
void parallel_for(int64_t n, const std::function<void(int64_t)>& fn) { pool().run(n, fn); }

void parallel_chunks(int64_t n, int64_t min_chunk, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    int64_t chunks = std::min<int64_t>(num_threads() * 4, std::max<int64_t>(1, n / std::max<int64_t>(1, min_chunk)));
    int64_t chunk = (n + chunks - 1) / chunks;
    parallel_for(chunks, [&](int64_t t) {
        int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b < e) fn(b, e);
    });
}

// ============================================================================
// GEMM: C = A * B^T, with both operands K-contiguous.
// 4x4 register tile; each of the 16 accumulators is a 4-lane vector that is
// horizontally summed at the end (portable to NEON / SSE via vector extensions).
// ============================================================================
namespace {

typedef float v4f __attribute__((vector_size(16), aligned(4)));

inline v4f ld4(const float* p) {
    v4f v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
inline float hsum(v4f v) { return (v[0] + v[1]) + (v[2] + v[3]); }

inline float dot(const float* a, const float* b, int64_t K) {
    v4f acc0 = {0, 0, 0, 0}, acc1 = {0, 0, 0, 0};
    int64_t k = 0;
    for (; k + 8 <= K; k += 8) {
        acc0 += ld4(a + k) * ld4(b + k);
        acc1 += ld4(a + k + 4) * ld4(b + k + 4);
    }
    float s = hsum(acc0) + hsum(acc1);
    for (; k < K; ++k) s += a[k] * b[k];
    return s;
}

void micro_4x4(int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc) {
    const float *a0 = A, *a1 = A + lda, *a2 = A + 2 * lda, *a3 = A + 3 * lda;
    const float *b0 = B, *b1 = B + ldb, *b2 = B + 2 * ldb, *b3 = B + 3 * ldb;
    v4f c00 = {0, 0, 0, 0}, c01 = c00, c02 = c00, c03 = c00;
    v4f c10 = c00, c11 = c00, c12 = c00, c13 = c00;
    v4f c20 = c00, c21 = c00, c22 = c00, c23 = c00;
    v4f c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        v4f x0 = ld4(a0 + k), x1 = ld4(a1 + k), x2 = ld4(a2 + k), x3 = ld4(a3 + k);
        v4f y0 = ld4(b0 + k), y1 = ld4(b1 + k), y2 = ld4(b2 + k), y3 = ld4(b3 + k);
        c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3;
        c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3;
        c20 += x2 * y0; c21 += x2 * y1; c22 += x2 * y2; c23 += x2 * y3;
        c30 += x3 * y0; c31 += x3 * y1; c32 += x3 * y2; c33 += x3 * y3;
    }
    float s[4][4] = {{hsum(c00), hsum(c01), hsum(c02), hsum(c03)},
                     {hsum(c10), hsum(c11), hsum(c12), hsum(c13)},
                     {hsum(c20), hsum(c21), hsum(c22), hsum(c23)},
                     {hsum(c30), hsum(c31), hsum(c32), hsum(c33)}};
    for (; k < K; ++k) {
        float x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
        float y0 = b0[k], y1 = b1[k], y2 = b2[k], y3 = b3[k];
        s[0][0] += x0 * y0; s[0][1] += x0 * y1; s[0][2] += x0 * y2; s[0][3] += x0 * y3;
        s[1][0] += x1 * y0; s[1][1] += x1 * y1; s[1][2] += x1 * y2; s[1][3] += x1 * y3;
        s[2][0] += x2 * y0; s[2][1] += x2 * y1; s[2][2] += x2 * y2; s[2][3] += x2 * y3;
        s[3][0] += x3 * y0; s[3][1] += x3 * y1; s[3][2] += x3 * y2; s[3][3] += x3 * y3;
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) C[i * ldc + j] += s[i][j];
}

void micro_edge(int64_t mr, int64_t nr, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb,
                float* C, int64_t ldc) {
    for (int64_t i = 0; i < mr; ++i)
        for (int64_t j = 0; j < nr; ++j) C[i * ldc + j] += dot(A + i * lda, B + j * ldb, K);
}

}  // namespace

void gemm_nt(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C,
             int64_t ldc, const float* bias, bool accumulate) {
    if (M <= 0 || N <= 0) return;
#ifdef USE_ACCELERATE
    float beta = accumulate ? 1.0f : 0.0f;
    if (bias) {
        for (int64_t i = 0; i < M; ++i) {
            float* c = C + i * ldc;
            if (accumulate)
                for (int64_t j = 0; j < N; ++j) c[j] += bias[j];
            else
                std::memcpy(c, bias, N * sizeof(float));
        }
        beta = 1.0f;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)M, (int)N, (int)K, 1.0f, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
    return;
#endif
    const int64_t MB = 64, NB = 128, KB = 128;
    const int64_t mblocks = (M + MB - 1) / MB, nblocks = (N + NB - 1) / NB;
    parallel_for(mblocks * nblocks, [&](int64_t t) {
        const int64_t m0 = (t / nblocks) * MB, m1 = std::min(M, m0 + MB);
        const int64_t n0 = (t % nblocks) * NB, n1 = std::min(N, n0 + NB);
        for (int64_t i = m0; i < m1; ++i) {
            float* c = C + i * ldc;
            if (!accumulate) {
                if (bias)
                    std::memcpy(c + n0, bias + n0, (n1 - n0) * sizeof(float));
                else
                    std::memset(c + n0, 0, (n1 - n0) * sizeof(float));
            } else if (bias) {
                for (int64_t j = n0; j < n1; ++j) c[j] += bias[j];
            }
        }
        for (int64_t k0 = 0; k0 < K; k0 += KB) {
            const int64_t kc = std::min(KB, K - k0);
            for (int64_t i = m0; i < m1; i += 4) {
                const int64_t mr = std::min<int64_t>(4, m1 - i);
                for (int64_t j = n0; j < n1; j += 4) {
                    const int64_t nr = std::min<int64_t>(4, n1 - j);
                    if (mr == 4 && nr == 4)
                        micro_4x4(kc, A + i * lda + k0, lda, B + j * ldb + k0, ldb, C + i * ldc + j, ldc);
                    else
                        micro_edge(mr, nr, kc, A + i * lda + k0, lda, B + j * ldb + k0, ldb, C + i * ldc + j, ldc);
                }
            }
        }
    });
}

// ============================================================================
// Layers
// ============================================================================
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b) {
    const int64_t in = x.dim(-1), out = W.dim(0);
    if (W.dim(1) != in) throw std::runtime_error("linear: input dim " + x.shape_str() + " vs weight " + W.shape_str());
    const int64_t rows = x.numel() / in;
    std::vector<int64_t> oshape = x.shape;
    oshape.back() = out;
    Tensor y(oshape);
    gemm_nt(rows, out, in, x.ptr(), in, W.ptr(), in, y.ptr(), out, b ? b->ptr() : nullptr, false);
    return y;
}

// im2col in chunks of output pixels so the scratch buffer stays around 64 MB,
// then out[Cout, pixels] = W[Cout, K] * cols[pixels, K]^T.
Tensor conv2d(const Tensor& x, const Tensor& W, const Tensor* b, int stride, int pad) {
    const int64_t N = x.dim(0), Cin = x.dim(1), H = x.dim(2), Wd = x.dim(3);
    const int64_t Cout = W.dim(0), kh = W.dim(2), kw = W.dim(3);
    if (W.dim(1) != Cin) throw std::runtime_error("conv2d: input " + x.shape_str() + " vs weight " + W.shape_str());
    const int64_t Ho = (H + 2 * pad - kh) / stride + 1, Wo = (Wd + 2 * pad - kw) / stride + 1;
    const int64_t K = Cin * kh * kw, HW = Ho * Wo;
    Tensor out({N, Cout, Ho, Wo});
    if (b) {  // bias is per output channel (a GEMM row), so seed the output with it
        for (int64_t n = 0; n < N; ++n)
            for (int64_t co = 0; co < Cout; ++co) {
                float* o = out.ptr() + (n * Cout + co) * HW;
                for (int64_t i = 0; i < HW; ++i) o[i] = b->ptr()[co];
            }
    }
    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(HW, (int64_t(1) << 24) / K));
    std::vector<float> cols((size_t)(chunk * K));
    for (int64_t n = 0; n < N; ++n) {
        const float* xin = x.ptr() + n * Cin * H * Wd;
        for (int64_t p0 = 0; p0 < HW; p0 += chunk) {
            const int64_t P = std::min(chunk, HW - p0);
            parallel_chunks(P, 32, [&](int64_t pb, int64_t pe) {
                for (int64_t p = pb; p < pe; ++p) {
                    const int64_t oy = (p0 + p) / Wo, ox = (p0 + p) % Wo;
                    float* row = cols.data() + p * K;
                    for (int64_t ci = 0; ci < Cin; ++ci) {
                        const float* xc = xin + ci * H * Wd;
                        for (int64_t ky = 0; ky < kh; ++ky) {
                            const int64_t iy = oy * stride - pad + ky;
                            float* r = row + (ci * kh + ky) * kw;
                            if (iy < 0 || iy >= H) {
                                for (int64_t kx = 0; kx < kw; ++kx) r[kx] = 0.0f;
                                continue;
                            }
                            const float* xr = xc + iy * Wd;
                            for (int64_t kx = 0; kx < kw; ++kx) {
                                const int64_t ix = ox * stride - pad + kx;
                                r[kx] = (ix >= 0 && ix < Wd) ? xr[ix] : 0.0f;
                            }
                        }
                    }
                }
            });
            gemm_nt(Cout, P, K, W.ptr(), K, cols.data(), K, out.ptr() + n * Cout * HW + p0, HW, nullptr,
                    /*accumulate=*/b != nullptr);
        }
    }
    return out;
}

Tensor group_norm(const Tensor& x, int groups, const Tensor& gamma, const Tensor& beta, float eps) {
    const int64_t N = x.dim(0), C = x.dim(1), HW = x.numel() / (N * C);
    if (C % groups != 0) throw std::runtime_error("group_norm: channels not divisible by groups");
    const int64_t cg = C / groups, len = cg * HW;
    Tensor y(x.shape);
    parallel_for(N * groups, [&](int64_t t) {
        const int64_t n = t / groups, g = t % groups;
        const float* p = x.ptr() + (n * C + g * cg) * HW;
        float* q = y.ptr() + (n * C + g * cg) * HW;
        double sum = 0;
        for (int64_t i = 0; i < len; ++i) sum += p[i];
        const double mean = sum / len;
        double var = 0;
        for (int64_t i = 0; i < len; ++i) {
            double d = p[i] - mean;
            var += d * d;
        }
        const float rstd = (float)(1.0 / std::sqrt(var / len + eps));
        for (int64_t c = 0; c < cg; ++c) {
            const int64_t ch = g * cg + c;
            const float a = rstd * gamma.ptr()[ch];
            const float bb = beta.ptr()[ch] - (float)mean * a;
            const float* pc = p + c * HW;
            float* qc = q + c * HW;
            for (int64_t i = 0; i < HW; ++i) qc[i] = pc[i] * a + bb;
        }
    });
    return y;
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    const int64_t C = x.dim(-1), rows = x.numel() / C;
    Tensor y(x.shape);
    parallel_chunks(rows, 8, [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; ++r) {
            const float* p = x.ptr() + r * C;
            float* q = y.ptr() + r * C;
            double sum = 0;
            for (int64_t i = 0; i < C; ++i) sum += p[i];
            const float mean = (float)(sum / C);
            double var = 0;
            for (int64_t i = 0; i < C; ++i) {
                float d = p[i] - mean;
                var += (double)d * d;
            }
            const float rstd = (float)(1.0 / std::sqrt(var / C + eps));
            for (int64_t i = 0; i < C; ++i) q[i] = (p[i] - mean) * rstd * gamma.ptr()[i] + beta.ptr()[i];
        }
    });
    return y;
}

void softmax_rows_(float* x, int64_t rows, int64_t cols, float scale, bool causal) {
    parallel_chunks(rows, 4, [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; ++r) {
            float* p = x + r * cols;
            const int64_t valid = causal ? std::min(cols, r + 1) : cols;
            float m = -INFINITY;
            for (int64_t j = 0; j < valid; ++j) m = std::max(m, p[j] * scale);
            float sum = 0.0f;
            for (int64_t j = 0; j < valid; ++j) {
                p[j] = std::exp(p[j] * scale - m);
                sum += p[j];
            }
            const float inv = 1.0f / sum;
            for (int64_t j = 0; j < valid; ++j) p[j] *= inv;
            for (int64_t j = valid; j < cols; ++j) p[j] = 0.0f;
        }
    });
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, bool causal) {
    const int64_t N = q.dim(0), Sq = q.dim(1), inner = q.dim(2), Skv = k.dim(1);
    if (inner % heads != 0) throw std::runtime_error("attention: inner dim not divisible by heads");
    if (k.dim(0) != N || v.dim(0) != N || v.dim(1) != Skv || k.dim(2) != inner || v.dim(2) != inner)
        throw std::runtime_error("attention: shape mismatch q" + q.shape_str() + " k" + k.shape_str());
    const int64_t d = inner / heads;
    const float scale = 1.0f / std::sqrt((float)d);
    Tensor out({N, Sq, inner});
    std::vector<float> scores((size_t)(Sq * Skv));
    std::vector<float> vT((size_t)(d * Skv));
    for (int64_t n = 0; n < N; ++n) {
        for (int h = 0; h < heads; ++h) {
            const float* qh = q.ptr() + n * Sq * inner + h * d;
            const float* kh = k.ptr() + n * Skv * inner + h * d;
            const float* vh = v.ptr() + n * Skv * inner + h * d;
            gemm_nt(Sq, Skv, d, qh, inner, kh, inner, scores.data(), Skv);
            softmax_rows_(scores.data(), Sq, Skv, scale, causal);
            // v_h^T [d, Skv] so that P V is again an NT product
            for (int64_t s = 0; s < Skv; ++s)
                for (int64_t j = 0; j < d; ++j) vT[j * Skv + s] = vh[s * inner + j];
            gemm_nt(Sq, d, Skv, scores.data(), Skv, vT.data(), Skv, out.ptr() + n * Sq * inner + h * d, inner);
        }
    }
    return out;
}

Tensor geglu(const Tensor& x) {
    const int64_t D2 = x.dim(-1), D = D2 / 2, rows = x.numel() / D2;
    std::vector<int64_t> oshape = x.shape;
    oshape.back() = D;
    Tensor y(oshape);
    parallel_chunks(rows, 8, [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; ++r) {
            const float* a = x.ptr() + r * D2;
            const float* g = a + D;
            float* o = y.ptr() + r * D;
            for (int64_t i = 0; i < D; ++i) {
                float gg = g[i];
                o[i] = a[i] * 0.5f * gg * (1.0f + std::erf(gg * 0.70710678118654752f));
            }
        }
    });
    return y;
}

// ============================================================================
// Elementwise / shape
// ============================================================================
void silu_(Tensor& x) {
    float* p = x.ptr();
    parallel_chunks(x.numel(), 16384, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) p[i] = p[i] / (1.0f + std::exp(-p[i]));
    });
}
void gelu_(Tensor& x) {
    float* p = x.ptr();
    parallel_chunks(x.numel(), 16384, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) p[i] = 0.5f * p[i] * (1.0f + std::erf(p[i] * 0.70710678118654752f));
    });
}
void quick_gelu_(Tensor& x) {
    float* p = x.ptr();
    parallel_chunks(x.numel(), 16384, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) p[i] = p[i] / (1.0f + std::exp(-1.702f * p[i]));
    });
}
void add_(Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) throw std::runtime_error("add_: " + a.shape_str() + " vs " + b.shape_str());
    float* p = a.ptr();
    const float* q = b.ptr();
    parallel_chunks(a.numel(), 16384, [&](int64_t s, int64_t e) {
        for (int64_t i = s; i < e; ++i) p[i] += q[i];
    });
}
void scale_(Tensor& x, float s) {
    for (float& v : x.data) v *= s;
}

Tensor upsample_nearest2x(const Tensor& x) {
    const int64_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    Tensor y({N, C, 2 * H, 2 * W});
    parallel_for(N * C, [&](int64_t nc) {
        const float* p = x.ptr() + nc * H * W;
        float* q = y.ptr() + nc * 4 * H * W;
        for (int64_t i = 0; i < H; ++i) {
            float* r0 = q + (2 * i) * 2 * W;
            float* r1 = r0 + 2 * W;
            for (int64_t j = 0; j < W; ++j) {
                float v = p[i * W + j];
                r0[2 * j] = r0[2 * j + 1] = v;
                r1[2 * j] = r1[2 * j + 1] = v;
            }
        }
    });
    return y;
}

Tensor concat_channels(const Tensor& a, const Tensor& b) {
    const int64_t N = a.dim(0), Ca = a.dim(1), Cb = b.dim(1), H = a.dim(2), W = a.dim(3);
    if (b.dim(0) != N || b.dim(2) != H || b.dim(3) != W)
        throw std::runtime_error("concat_channels: " + a.shape_str() + " vs " + b.shape_str());
    Tensor y({N, Ca + Cb, H, W});
    const int64_t HW = H * W;
    for (int64_t n = 0; n < N; ++n) {
        std::memcpy(y.ptr() + n * (Ca + Cb) * HW, a.ptr() + n * Ca * HW, Ca * HW * sizeof(float));
        std::memcpy(y.ptr() + (n * (Ca + Cb) + Ca) * HW, b.ptr() + n * Cb * HW, Cb * HW * sizeof(float));
    }
    return y;
}

Tensor concat_batch(const Tensor& a, const Tensor& b) {
    std::vector<int64_t> s = a.shape;
    s[0] += b.dim(0);
    Tensor y(s);
    std::memcpy(y.ptr(), a.ptr(), a.numel() * sizeof(float));
    std::memcpy(y.ptr() + a.numel(), b.ptr(), b.numel() * sizeof(float));
    return y;
}

Tensor nchw_to_nlc(const Tensor& x) {
    const int64_t N = x.dim(0), C = x.dim(1), HW = x.dim(2) * x.dim(3);
    Tensor y({N, HW, C});
    parallel_for(N * C, [&](int64_t nc) {
        const int64_t n = nc / C, c = nc % C;
        const float* p = x.ptr() + nc * HW;
        float* q = y.ptr() + n * HW * C + c;
        for (int64_t i = 0; i < HW; ++i) q[i * C] = p[i];
    });
    return y;
}

Tensor nlc_to_nchw(const Tensor& x, int64_t H, int64_t W) {
    const int64_t N = x.dim(0), HW = x.dim(1), C = x.dim(2);
    if (HW != H * W) throw std::runtime_error("nlc_to_nchw: bad spatial size");
    Tensor y({N, C, H, W});
    parallel_for(N * C, [&](int64_t nc) {
        const int64_t n = nc / C, c = nc % C;
        const float* p = x.ptr() + n * HW * C + c;
        float* q = y.ptr() + nc * HW;
        for (int64_t i = 0; i < HW; ++i) q[i] = p[i * C];
    });
    return y;
}

Tensor slice_batch(const Tensor& x, int64_t b) {
    std::vector<int64_t> s = x.shape;
    s[0] = 1;
    Tensor y(s);
    const int64_t per = x.numel() / x.dim(0);
    std::memcpy(y.ptr(), x.ptr() + b * per, per * sizeof(float));
    return y;
}

Tensor repeat_batch(const Tensor& x, int64_t times) {
    std::vector<int64_t> s = x.shape;
    s[0] = x.dim(0) * times;
    Tensor y(s);
    for (int64_t t = 0; t < times; ++t) std::memcpy(y.ptr() + t * x.numel(), x.ptr(), x.numel() * sizeof(float));
    return y;
}

}  // namespace sd
