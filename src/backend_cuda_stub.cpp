// ============================================================================
// backend_cuda_stub.cpp - Stands in for backend_cuda.cu when the project is
// built without CUDA (the default). available() is false; nothing else
// should be reached, and every call throws so a mistake is loud.
// ============================================================================
#include <stdexcept>

#include "backend_cuda.h"

namespace sd {
namespace cuda {

namespace {
[[noreturn]] void unavailable() { throw std::runtime_error("CUDA backend not built (make CUDA=1) or no device"); }
}  // namespace

bool available() { return false; }
const char* device_name() { return ""; }
size_t device_memory() { return 0; }

void* malloc(size_t) { unavailable(); }
void free(void*) {}
void upload(void*, const void*, size_t) { unavailable(); }
void download(void*, const void*, size_t) { unavailable(); }
void copy(void*, const void*, size_t) { unavailable(); }
void zero(void*, size_t) { unavailable(); }
void synchronize() {}

void gemm_nt_f32(int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, float*, int64_t, const float*, bool) { unavailable(); }
void gemm_nt_bf16(int64_t, int64_t, int64_t, const float*, int64_t, const uint16_t*, int64_t, float*, int64_t, const float*, bool) { unavailable(); }
void conv2d(const float*, int64_t, int64_t, int64_t, int64_t, const float*, int64_t, int64_t, int64_t, const float*, int, int, float*, int64_t, int64_t) { unavailable(); }
void group_norm(const float*, float*, int64_t, int64_t, int64_t, int, const float*, const float*, float) { unavailable(); }
void layer_norm(const float*, float*, int64_t, int64_t, const float*, const float*, float) { unavailable(); }
void rms_norm(const float*, float*, int64_t, int64_t, const float*, float) { unavailable(); }
void rms_norm_heads(float*, int64_t, int64_t, const float*, float) { unavailable(); }
void rope(float*, int64_t, int64_t, int64_t, int64_t, const float*, const float*, bool) { unavailable(); }
void softmax_rows(float*, int64_t, int64_t, float, bool) { unavailable(); }
void attention(const float*, const float*, const float*, float*, int64_t, int64_t, int64_t, int, int, int64_t, bool) { unavailable(); }
void geglu(const float*, float*, int64_t, int64_t) { unavailable(); }
void swiglu(const float*, const float*, float*, int64_t) { unavailable(); }
void silu(float*, int64_t) { unavailable(); }
void gelu(float*, int64_t) { unavailable(); }
void quick_gelu(float*, int64_t) { unavailable(); }
void add(float*, const float*, int64_t) { unavailable(); }
void scale(float*, float, int64_t) { unavailable(); }
void affine(float*, float, float, int64_t) { unavailable(); }
void mul_rows(float*, const float*, int64_t, int64_t) { unavailable(); }
void add_rows_scaled(float*, const float*, const float*, int64_t, int64_t) { unavailable(); }
void add_channel_bias(float*, const float*, int64_t, int64_t, int64_t) { unavailable(); }
void upsample_nearest2x(const float*, float*, int64_t, int64_t, int64_t) { unavailable(); }
void concat_channels(const float*, const float*, float*, int64_t, int64_t, int64_t, int64_t) { unavailable(); }
void nchw_to_nlc(const float*, float*, int64_t, int64_t, int64_t) { unavailable(); }
void nlc_to_nchw(const float*, float*, int64_t, int64_t, int64_t) { unavailable(); }
void patchify(const float*, float*, int64_t, int64_t, int64_t, int) { unavailable(); }
void unpatchify(const float*, float*, int64_t, int64_t, int64_t, int) { unavailable(); }

}  // namespace cuda
}  // namespace sd
