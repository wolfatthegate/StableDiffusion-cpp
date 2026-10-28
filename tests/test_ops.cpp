// ============================================================================
// test_ops.cpp - Checks the hand-written kernels against naive reference
// implementations (computed in double) on random inputs, then benchmarks one
// representative GEMM. Odd sizes and padded leading dimensions are used on
// purpose to exercise the edge-tile and stride code paths.
// Exit status is non-zero if any check fails. Run with `make test`.
// ============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "backend_cuda.h"
#include "tensor.h"

using namespace sd;

static int failures = 0;

// Prints one result line and counts failures.
static void check(const char* name, double max_err, double tol) {
    bool ok = max_err <= tol;
    std::printf("%-28s max|err| = %.3e  %s\n", name, max_err, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

static void fill(std::vector<float>& v, std::mt19937& rng) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (float& x : v) x = n(rng);
}

// gemm_nt with leading dimensions larger than the logical sizes, optional
// bias and accumulate, against a triple loop.
static void test_gemm(std::mt19937& rng, int M, int N, int K, bool bias, bool acc) {
    std::vector<float> A((size_t)M * (K + 3)), B((size_t)N * (K + 5)), C((size_t)M * (N + 2)), ref, bv(N);
    fill(A, rng); fill(B, rng); fill(C, rng); fill(bv, rng);
    ref = C;
    const int lda = K + 3, ldb = K + 5, ldc = N + 2;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = acc ? ref[i * ldc + j] : 0.0;
            if (bias) s += bv[j];
            for (int k = 0; k < K; ++k) s += (double)A[i * lda + k] * B[j * ldb + k];
            ref[i * ldc + j] = (float)s;
        }
    gemm_nt(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, bias ? bv.data() : nullptr, acc);
    double err = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) err = std::max(err, (double)std::fabs(C[i * ldc + j] - ref[i * ldc + j]));
    char name[64];
    std::snprintf(name, sizeof name, "gemm %dx%dx%d%s%s", M, N, K, bias ? " +bias" : "", acc ? " acc" : "");
    check(name, err, 1e-3 * std::sqrt((double)K));
}

// conv2d against a direct 7-nested-loop convolution with zero padding.
static void test_conv(std::mt19937& rng, int N, int Cin, int Cout, int H, int W, int k, int stride) {
    Tensor x({N, Cin, H, W}), w({Cout, Cin, k, k}), b({Cout});
    fill(x.data, rng); fill(w.data, rng); fill(b.data, rng);
    const int pad = (k - 1) / 2;
    Tensor y = conv2d(x, w, &b, stride, pad);
    const int Ho = (int)y.dim(2), Wo = (int)y.dim(3);
    double err = 0;
    for (int n = 0; n < N; ++n)
        for (int co = 0; co < Cout; ++co)
            for (int oy = 0; oy < Ho; ++oy)
                for (int ox = 0; ox < Wo; ++ox) {
                    double s = b.ptr()[co];
                    for (int ci = 0; ci < Cin; ++ci)
                        for (int ky = 0; ky < k; ++ky)
                            for (int kx = 0; kx < k; ++kx) {
                                int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                                s += (double)x.ptr()[((n * Cin + ci) * H + iy) * W + ix] *
                                     w.ptr()[((co * Cin + ci) * k + ky) * k + kx];
                            }
                    err = std::max(err, std::fabs(s - y.ptr()[((n * Cout + co) * Ho + oy) * Wo + ox]));
                }
    char name[64];
    std::snprintf(name, sizeof name, "conv2d %dx%d k%d s%d", Cin, Cout, k, stride);
    check(name, err, 1e-3 * std::sqrt((double)Cin * k * k));
}

// attention against a per-head scalar softmax(QK^T/sqrt(d))V.
static void test_attention(std::mt19937& rng, int N, int Sq, int Skv, int heads, int d, bool causal) {
    Tensor q({N, Sq, heads * d}), k({N, Skv, heads * d}), v({N, Skv, heads * d});
    fill(q.data, rng); fill(k.data, rng); fill(v.data, rng);
    Tensor o = attention(q, k, v, heads, causal);
    const int inner = heads * d;
    const float scale = 1.0f / std::sqrt((float)d);
    double err = 0;
    std::vector<double> sc(Skv);
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < heads; ++h)
            for (int i = 0; i < Sq; ++i) {
                double m = -1e30;
                for (int j = 0; j < Skv; ++j) {
                    double s = 0;
                    for (int t = 0; t < d; ++t)
                        s += (double)q.ptr()[(n * Sq + i) * inner + h * d + t] * k.ptr()[(n * Skv + j) * inner + h * d + t];
                    sc[j] = (causal && j > i) ? -1e30 : s * scale;
                    m = std::max(m, sc[j]);
                }
                double sum = 0;
                for (int j = 0; j < Skv; ++j) { sc[j] = (causal && j > i) ? 0.0 : std::exp(sc[j] - m); sum += sc[j]; }
                for (int t = 0; t < d; ++t) {
                    double acc = 0;
                    for (int j = 0; j < Skv; ++j) acc += sc[j] / sum * v.ptr()[(n * Skv + j) * inner + h * d + t];
                    err = std::max(err, std::fabs(acc - o.ptr()[(n * Sq + i) * inner + h * d + t]));
                }
            }
    char name[64];
    std::snprintf(name, sizeof name, "attention h%d d%d %s", heads, d, causal ? "causal" : "");
    check(name, err, 1e-4);
}

// group_norm and layer_norm with identity affine: outputs must have mean 0
// and variance 1 per normalisation group.
static void test_norms(std::mt19937& rng) {
    Tensor x({2, 64, 5, 7}), g({64}), b({64});
    fill(x.data, rng);
    for (float& v : x.data) v = v * 3.0f + 2.0f;
    for (int i = 0; i < 64; ++i) { g.ptr()[i] = 1.0f; b.ptr()[i] = 0.0f; }
    Tensor y = group_norm(x, 32, g, b, 1e-6f);
    double err = 0;
    const int cg = 2, HW = 35;
    for (int n = 0; n < 2; ++n)
        for (int gi = 0; gi < 32; ++gi) {
            double mean = 0, var = 0;
            const float* p = y.ptr() + (n * 64 + gi * cg) * HW;
            for (int i = 0; i < cg * HW; ++i) mean += p[i];
            mean /= cg * HW;
            for (int i = 0; i < cg * HW; ++i) var += (p[i] - mean) * (p[i] - mean);
            var /= cg * HW;
            err = std::max(err, std::max(std::fabs(mean), std::fabs(var - 1.0)));
        }
    check("group_norm (mean0/var1)", err, 1e-3);

    Tensor z({3, 7, 64});
    fill(z.data, rng);
    Tensor ln = layer_norm(z, g, b, 1e-5f);
    err = 0;
    for (int r = 0; r < 21; ++r) {
        double mean = 0, var = 0;
        const float* p = ln.ptr() + r * 64;
        for (int i = 0; i < 64; ++i) mean += p[i];
        mean /= 64;
        for (int i = 0; i < 64; ++i) var += (p[i] - mean) * (p[i] - mean);
        var /= 64;
        err = std::max(err, std::max(std::fabs(mean), std::fabs(var - 1.0)));
    }
    check("layer_norm (mean0/var1)", err, 1e-3);
}

// Times a GEMM shaped like a 3x3 conv on 64x64 latents with 320 channels.
// gemm_nt_bf16 against gemm_nt on the same (bf16-rounded) B.
static void test_gemm_bf16(std::mt19937& rng, int M, int N, int K) {
    std::vector<float> A((size_t)M * K), Bf((size_t)N * K), C1((size_t)M * N), C2((size_t)M * N), bias(N);
    std::vector<uint16_t> Bh((size_t)N * K);
    fill(A, rng); fill(Bf, rng); fill(bias, rng);
    for (size_t i = 0; i < Bf.size(); ++i) { Bh[i] = f32_to_bf16(Bf[i]); Bf[i] = bf16_to_f32(Bh[i]); }
    gemm_nt(M, N, K, A.data(), K, Bf.data(), K, C1.data(), N, bias.data());
    gemm_nt_bf16(M, N, K, A.data(), K, Bh.data(), K, C2.data(), N, bias.data());
    double err = 0;
    for (size_t i = 0; i < C1.size(); ++i) err = std::max(err, (double)std::fabs(C1[i] - C2[i]));
    char name[64];
    std::snprintf(name, sizeof name, "gemm_bf16 %dx%dx%d", M, N, K);
    check(name, err, 1e-4 * std::sqrt((double)K));
}

// bf16 round trip and rounding direction.
static void test_bf16() {
    double err = 0;
    for (float v : {0.0f, 1.0f, -1.5f, 3.14159f, 1e-3f, 65504.0f, 1e-30f}) {
        float r = bf16_to_f32(f32_to_bf16(v));
        err = std::max(err, (double)std::fabs(r - v) / (std::fabs(v) + 1e-30));
    }
    check("bf16 round trip (rel)", err, 1.0 / 128);  // 8 bits of mantissa
}

static void test_rms_rope_swiglu(std::mt19937& rng) {
    const int S = 5, H = 3, D = 8;
    Tensor x({S, H * D}), g({H * D}), gd({D});
    fill(x.data, rng); fill(g.data, rng); fill(gd.data, rng);
    Tensor y = rms_norm(x, &g, 1e-6f);
    double err = 0;
    for (int r = 0; r < S; ++r) {
        double ss = 0;
        for (int i = 0; i < H * D; ++i) ss += (double)x.ptr()[r * H * D + i] * x.ptr()[r * H * D + i];
        double rstd = 1.0 / std::sqrt(ss / (H * D) + 1e-6);
        for (int i = 0; i < H * D; ++i) err = std::max(err, std::fabs(x.ptr()[r * H * D + i] * rstd * g.ptr()[i] - y.ptr()[r * H * D + i]));
    }
    check("rms_norm", err, 1e-5);

    Tensor xh = x;
    rms_norm_heads_(xh, D, &gd, 1e-6f);
    err = 0;
    for (int r = 0; r < S * H; ++r) {
        double ss = 0;
        for (int i = 0; i < D; ++i) ss += (double)x.ptr()[r * D + i] * x.ptr()[r * D + i];
        double rstd = 1.0 / std::sqrt(ss / D + 1e-6);
        for (int i = 0; i < D; ++i) err = std::max(err, std::fabs(x.ptr()[r * D + i] * rstd * gd.ptr()[i] - xh.ptr()[r * D + i]));
    }
    check("rms_norm_heads_", err, 1e-5);

    // RoPE, rotate_half convention: pair (i, i + D/2) rotated by angle theta_i.
    Tensor cs({S, D}), sn({S, D});
    std::vector<double> ang((size_t)S * D / 2);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < D / 2; ++i) {
            double a = s * std::pow(10000.0, -2.0 * i / D);
            ang[s * D / 2 + i] = a;
            cs.ptr()[s * D + i] = cs.ptr()[s * D + i + D / 2] = (float)std::cos(a);
            sn.ptr()[s * D + i] = sn.ptr()[s * D + i + D / 2] = (float)std::sin(a);
        }
    Tensor xr = x;
    rope_(xr, D, cs, sn, false);
    err = 0;
    for (int s = 0; s < S; ++s)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < D / 2; ++i) {
                double a = x.ptr()[(s * H + h) * D + i], b = x.ptr()[(s * H + h) * D + i + D / 2], t = ang[s * D / 2 + i];
                err = std::max(err, std::fabs(a * std::cos(t) - b * std::sin(t) - xr.ptr()[(s * H + h) * D + i]));
                err = std::max(err, std::fabs(b * std::cos(t) + a * std::sin(t) - xr.ptr()[(s * H + h) * D + i + D / 2]));
            }
    check("rope rotate_half", err, 1e-5);

    // RoPE, interleaved convention: pair (2i, 2i+1); tables hold each angle twice.
    Tensor cs2({S, D}), sn2({S, D});
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < D / 2; ++i) {
            double a = ang[s * D / 2 + i];
            cs2.ptr()[s * D + 2 * i] = cs2.ptr()[s * D + 2 * i + 1] = (float)std::cos(a);
            sn2.ptr()[s * D + 2 * i] = sn2.ptr()[s * D + 2 * i + 1] = (float)std::sin(a);
        }
    Tensor xi = x;
    rope_(xi, D, cs2, sn2, true);
    err = 0;
    for (int s = 0; s < S; ++s)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < D / 2; ++i) {
                double a = x.ptr()[(s * H + h) * D + 2 * i], b = x.ptr()[(s * H + h) * D + 2 * i + 1], t = ang[s * D / 2 + i];
                err = std::max(err, std::fabs(a * std::cos(t) - b * std::sin(t) - xi.ptr()[(s * H + h) * D + 2 * i]));
                err = std::max(err, std::fabs(b * std::cos(t) + a * std::sin(t) - xi.ptr()[(s * H + h) * D + 2 * i + 1]));
            }
    check("rope interleaved", err, 1e-5);

    Tensor u({S, H * D});
    fill(u.data, rng);
    Tensor sw = swiglu(x, u);
    err = 0;
    for (int i = 0; i < S * H * D; ++i) {
        double gv = x.ptr()[i];
        err = std::max(err, std::fabs(gv / (1.0 + std::exp(-gv)) * u.ptr()[i] - sw.ptr()[i]));
    }
    check("swiglu", err, 1e-5);
}

// Grouped-query attention: 4 query heads sharing 2 key/value heads.
static void test_gqa(std::mt19937& rng) {
    const int N = 1, S = 6, heads = 4, kv = 2, d = 4;
    Tensor q({N, S, heads * d}), k({N, S, kv * d}), v({N, S, kv * d});
    fill(q.data, rng); fill(k.data, rng); fill(v.data, rng);
    Tensor o = attention(q, k, v, heads, true, kv);
    // Expand k/v to all heads and compare with the plain path.
    Tensor kx({N, S, heads * d}), vx({N, S, heads * d});
    for (int s = 0; s < S; ++s)
        for (int h = 0; h < heads; ++h)
            for (int t = 0; t < d; ++t) {
                kx.ptr()[s * heads * d + h * d + t] = k.ptr()[s * kv * d + (h / (heads / kv)) * d + t];
                vx.ptr()[s * heads * d + h * d + t] = v.ptr()[s * kv * d + (h / (heads / kv)) * d + t];
            }
    Tensor o2 = attention(q, kx, vx, heads, true);
    double err = 0;
    for (int i = 0; i < o.numel(); ++i) err = std::max(err, (double)std::fabs(o.ptr()[i] - o2.ptr()[i]));
    check("attention gqa causal", err, 1e-6);
}

static void bench_gemm() {
    const int M = 4096, N = 320, K = 2880;  // a typical UNet conv GEMM at 64x64 latents
    std::vector<float> A((size_t)M * K, 0.5f), B((size_t)N * K, 0.25f), C((size_t)M * N);
    gemm_nt(M, N, K, A.data(), K, B.data(), K, C.data(), N);  // warm-up
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 5;
    for (int r = 0; r < reps; ++r) gemm_nt(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("gemm %dx%dx%d: %.1f ms  %.1f GFLOP/s on %d threads\n", M, N, K, s * 1e3, 2.0 * M * N * K / s / 1e9,
                num_threads());
}

static void bench_gemm_bf16() {
    const int M = 4096, N = 3840, K = 3840;  // one Z-Image attention projection at 1024x1024
    std::vector<float> A((size_t)M * K, 0.5f), C((size_t)M * N);
    std::vector<uint16_t> B((size_t)N * K, f32_to_bf16(0.25f));
    gemm_nt_bf16(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 3;
    for (int r = 0; r < reps; ++r) gemm_nt_bf16(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("gemm_bf16 %dx%dx%d: %.1f ms  %.1f GFLOP/s\n", M, N, K, s * 1e3, 2.0 * M * N * K / s / 1e9);
}

// ---------------------------------------------------------------------------
// CUDA backend: every op is run on the CPU and on the GPU with the same
// inputs and the results compared. Skipped when no CUDA device is available.
// ---------------------------------------------------------------------------
static Tensor randn_t(std::vector<int64_t> shape, std::mt19937& rng) {
    Tensor t(std::move(shape));
    fill(t.data, rng);
    return t;
}

static double max_diff(const Tensor& a, const Tensor& b_any) {
    Tensor b = b_any.to(Device::CPU);
    if (a.numel() != b.numel()) return 1e30;
    double err = 0;
    for (int64_t i = 0; i < a.numel(); ++i) err = std::max(err, (double)std::fabs(a.ptr()[i] - b.ptr()[i]));
    return err;
}

static void cuda_check(const char* name, const Tensor& cpu, const Tensor& gpu, double tol) {
    char label[64];
    std::snprintf(label, sizeof label, "cuda %s", name);
    check(label, max_diff(cpu, gpu), tol);
}

static void test_cuda(std::mt19937& rng) {
    if (!cuda::available()) {
        std::printf("cuda: not available, GPU comparisons skipped\n");
        return;
    }
    std::printf("cuda: %s\n", cuda::device_name());
    const Device G = Device::CUDA;

    // round trip
    {
        Tensor x = randn_t({3, 70}, rng);
        cuda_check("to(cuda).to(cpu)", x, x.to(G), 0.0);
    }
    // linear, f32 and bf16 weights, with bias
    {
        Tensor x = randn_t({2, 37, 300}, rng), W = randn_t({130, 300}, rng), b = randn_t({130}, rng);
        Tensor y = linear(x, W, &b);
        cuda_check("linear f32", y, linear(x.to(G), W.to(G), nullptr), 1e30);  // shape probe
        Tensor bg = b.to(G);
        cuda_check("linear f32 +bias", y, linear(x.to(G), W.to(G), &bg), 1e-3);
        Tensor Wh = W;
        Wh.to_bf16();
        Tensor yh = linear(x, Wh, &b);
        cuda_check("linear bf16 +bias", yh, linear(x.to(G), Wh.to(G), &bg), 3e-2);  // A is rounded to bf16 on the GPU
    }
    // conv2d
    {
        Tensor x = randn_t({2, 5, 9, 11}, rng), w = randn_t({6, 5, 3, 3}, rng), b = randn_t({6}, rng), bg = b.to(G);
        cuda_check("conv2d k3 s1", conv2d(x, w, &b, 1, 1), conv2d(x.to(G), w.to(G), &bg, 1, 1), 1e-3);
        cuda_check("conv2d k3 s2", conv2d(x, w, &b, 2, 1), conv2d(x.to(G), w.to(G), &bg, 2, 1), 1e-3);
        Tensor w1 = randn_t({4, 5, 1, 1}, rng), b1 = randn_t({4}, rng), b1g = b1.to(G);
        cuda_check("conv2d k1", conv2d(x, w1, &b1, 1, 0), conv2d(x.to(G), w1.to(G), &b1g, 1, 0), 1e-3);
    }
    // norms
    {
        Tensor x = randn_t({2, 64, 5, 7}, rng), g = randn_t({64}, rng), b = randn_t({64}, rng);
        cuda_check("group_norm", group_norm(x, 32, g, b, 1e-6f), group_norm(x.to(G), 32, g.to(G), b.to(G), 1e-6f), 1e-4);
        Tensor z = randn_t({3, 7, 64}, rng);
        cuda_check("layer_norm", layer_norm(z, g, b, 1e-5f), layer_norm(z.to(G), g.to(G), b.to(G), 1e-5f), 1e-4);
        cuda_check("layer_norm no affine", layer_norm(z, nullptr, nullptr, 1e-5f), layer_norm(z.to(G), nullptr, nullptr, 1e-5f), 1e-4);
        Tensor gG = g.to(G);
        cuda_check("rms_norm", rms_norm(z, &g, 1e-6f), rms_norm(z.to(G), &gG, 1e-6f), 1e-4);
        Tensor zh = z, zg = z.to(G), gd = randn_t({8}, rng), gdg = gd.to(G);
        rms_norm_heads_(zh, 8, &gd, 1e-6f);
        rms_norm_heads_(zg, 8, &gdg, 1e-6f);
        cuda_check("rms_norm_heads_", zh, zg, 1e-4);
    }
    // rope (both conventions), swiglu, geglu, elementwise
    {
        const int S = 5, H = 3, D = 8;
        Tensor x = randn_t({1, S, H * D}, rng), cs({S, D}), sn({S, D});
        for (int s = 0; s < S; ++s)
            for (int i = 0; i < D; ++i) {
                cs.ptr()[s * D + i] = std::cos(0.1f * s * (i % (D / 2) + 1));
                sn.ptr()[s * D + i] = std::sin(0.1f * s * (i % (D / 2) + 1));
            }
        for (bool inter : {false, true}) {
            Tensor a = x, b = x.to(G);
            rope_(a, D, cs, sn, inter);
            rope_(b, D, cs.to(G), sn.to(G), inter);
            cuda_check(inter ? "rope interleaved" : "rope rotate_half", a, b, 1e-5);
        }
        Tensor u = randn_t({1, S, H * D}, rng);
        cuda_check("swiglu", swiglu(x, u), swiglu(x.to(G), u.to(G)), 1e-5);
        Tensor gg = randn_t({6, 40}, rng);
        cuda_check("geglu", geglu(gg), geglu(gg.to(G)), 1e-5);
        for (int k = 0; k < 3; ++k) {
            Tensor a = randn_t({1000}, rng), b = a.to(G);
            if (k == 0) { silu_(a); silu_(b); }
            if (k == 1) { gelu_(a); gelu_(b); }
            if (k == 2) { quick_gelu_(a); quick_gelu_(b); }
            cuda_check(k == 0 ? "silu_" : k == 1 ? "gelu_" : "quick_gelu_", a, b, 1e-5);
        }
        Tensor a = randn_t({1000}, rng), b = randn_t({1000}, rng), ag = a.to(G);
        add_(a, b); add_(ag, b.to(G));
        cuda_check("add_", a, ag, 1e-6);
        scale_(a, 0.37f); scale_(ag, 0.37f);
        cuda_check("scale_", a, ag, 1e-6);
        affine_(a, 1.5f, -0.25f); affine_(ag, 1.5f, -0.25f);
        cuda_check("affine_", a, ag, 1e-5);
    }
    // attention: MHA, GQA, causal
    {
        Tensor q = randn_t({2, 9, 24}, rng), k = randn_t({2, 9, 24}, rng), v = randn_t({2, 9, 24}, rng);
        cuda_check("attention h3", attention(q, k, v, 3), attention(q.to(G), k.to(G), v.to(G), 3), 1e-4);
        cuda_check("attention h3 causal", attention(q, k, v, 3, true), attention(q.to(G), k.to(G), v.to(G), 3, true), 1e-4);
        Tensor kk = randn_t({2, 9, 8}, rng), vv = randn_t({2, 9, 8}, rng);
        cuda_check("attention gqa", attention(q, kk, vv, 3, false, 1), attention(q.to(G), kk.to(G), vv.to(G), 3, false, 1), 1e-4);
        Tensor qb = randn_t({1, 100, 64}, rng), kb = randn_t({1, 100, 64}, rng), vb = randn_t({1, 100, 64}, rng);
        cuda_check("attention 100 tokens", attention(qb, kb, vb, 2), attention(qb.to(G), kb.to(G), vb.to(G), 2), 1e-4);
    }
    // row helpers and shape ops
    {
        Tensor x = randn_t({4, 6}, rng), v = randn_t({6}, rng), y = randn_t({4, 6}, rng);
        Tensor xg = x.to(G);
        mul_rows_(x, v); mul_rows_(xg, v.to(G));
        cuda_check("mul_rows_", x, xg, 1e-6);
        add_rows_scaled_(x, y, v); add_rows_scaled_(xg, y.to(G), v.to(G));
        cuda_check("add_rows_scaled_", x, xg, 1e-6);
        Tensor img = randn_t({2, 3, 4, 5}, rng), bias = randn_t({2, 3}, rng), imgg = img.to(G);
        add_channel_bias_(img, bias); add_channel_bias_(imgg, bias.to(G));
        cuda_check("add_channel_bias_", img, imgg, 1e-6);
        cuda_check("upsample_nearest2x", upsample_nearest2x(img), upsample_nearest2x(img.to(G)), 0.0);
        Tensor img2 = randn_t({2, 2, 4, 5}, rng);
        cuda_check("concat_channels", concat_channels(img, img2), concat_channels(img.to(G), img2.to(G)), 0.0);
        cuda_check("concat_batch", concat_batch(img, img2), concat_batch(img.to(G), img2.to(G)), 0.0);
        cuda_check("nchw_to_nlc", nchw_to_nlc(img), nchw_to_nlc(img.to(G)), 0.0);
        cuda_check("nlc_to_nchw", nlc_to_nchw(nchw_to_nlc(img), 4, 5), nlc_to_nchw(nchw_to_nlc(img.to(G)), 4, 5), 0.0);
        cuda_check("slice_batch", slice_batch(img, 1), slice_batch(img.to(G), 1), 0.0);
        cuda_check("repeat_batch", repeat_batch(img, 3), repeat_batch(img.to(G), 3), 0.0);
        cuda_check("slice_rows", slice_rows(x, 1, 3), slice_rows(xg, 1, 3), 0.0);
        Tensor lat = randn_t({1, 16, 6, 8}, rng);
        cuda_check("patchify2d", patchify2d(lat, 2), patchify2d(lat.to(G), 2), 0.0);
        Tensor tok = patchify2d(lat, 2);
        cuda_check("unpatchify2d", unpatchify2d(tok, 16, 6, 8, 2), unpatchify2d(tok.to(G), 16, 6, 8, 2), 0.0);
    }
}

int main() {
    std::mt19937 rng(1234);
    test_gemm(rng, 4, 4, 4, false, false);
    test_gemm(rng, 37, 53, 129, true, false);
    test_gemm(rng, 70, 130, 300, false, true);
    test_gemm(rng, 1, 77, 40, true, false);
    test_gemm(rng, 129, 5, 1000, true, true);
    test_conv(rng, 2, 3, 5, 9, 11, 3, 1);
    test_conv(rng, 1, 4, 6, 8, 8, 3, 2);
    test_conv(rng, 1, 7, 3, 5, 6, 1, 1);
    test_attention(rng, 2, 9, 9, 3, 8, false);
    test_attention(rng, 1, 11, 5, 2, 4, false);
    test_attention(rng, 1, 7, 7, 1, 16, true);
    test_norms(rng);
    test_bf16();
    test_gemm_bf16(rng, 37, 53, 129);
    test_gemm_bf16(rng, 130, 257, 300);
    test_rms_rope_swiglu(rng);
    test_gqa(rng);
    test_cuda(rng);
    bench_gemm();
    bench_gemm_bf16();
    std::printf(failures ? "\n%d FAILED\n" : "\nall ok\n", failures);
    return failures ? 1 : 0;
}
