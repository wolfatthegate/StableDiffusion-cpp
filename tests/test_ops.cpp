// Checks the hand-written kernels against naive reference implementations.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "tensor.h"

using namespace sd;

static int failures = 0;

static void check(const char* name, double max_err, double tol) {
    bool ok = max_err <= tol;
    std::printf("%-28s max|err| = %.3e  %s\n", name, max_err, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

static void fill(std::vector<float>& v, std::mt19937& rng) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (float& x : v) x = n(rng);
}

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
    bench_gemm();
    std::printf(failures ? "\n%d FAILED\n" : "\nall ok\n", failures);
    return failures ? 1 : 0;
}
