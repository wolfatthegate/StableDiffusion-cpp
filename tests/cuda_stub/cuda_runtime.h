// Minimal stand-in for the CUDA runtime header, used only to syntax-check
// src/backend_cuda.cu with clang on machines without the CUDA toolkit:
//   make cuda-syntax-check
// It declares just what the backend uses. Not a real implementation.
#pragma once
#include <cmath>
#include <cstddef>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#include <__clang_cuda_builtin_vars.h>
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
enum cudaError_t { cudaSuccess = 0, cudaErrorOther = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
typedef void* cudaStream_t;
struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; };
extern "C" cudaError_t cudaConfigureCall(dim3 g, dim3 b, size_t s = 0, cudaStream_t st = 0);
extern "C" cudaError_t __cudaPushCallConfiguration(dim3 g, dim3 b, size_t s = 0, cudaStream_t st = 0);
cudaError_t cudaMalloc(void** p, size_t n);
cudaError_t cudaFree(void* p);
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k);
cudaError_t cudaMemset(void* d, int v, size_t n);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t e);
cudaError_t cudaGetDeviceCount(int* n);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int dev);
extern "C" __device__ void __syncthreads();
__device__ float expf(float);
__device__ float erff(float);
__device__ float fmaxf(float, float);
__device__ float sqrtf(float);
__device__ double sqrt(double);
__device__ unsigned __float_as_uint(float);
__device__ float __uint_as_float(unsigned);
__device__ long long min(long long, long long);
