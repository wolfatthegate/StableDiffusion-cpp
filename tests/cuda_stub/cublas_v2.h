// Minimal stand-in for cublas_v2.h for the clang syntax check (see cuda_runtime.h).
#pragma once
struct cublasContext;
typedef cublasContext* cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
enum cublasMath_t { CUBLAS_DEFAULT_MATH, CUBLAS_TF32_TENSOR_OP_MATH };
enum cudaDataType { CUDA_R_16BF, CUDA_R_32F };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT };
cublasStatus_t cublasCreate(cublasHandle_t* h);
cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m);
cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float* alpha,
                           const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                         const float* alpha, const float* A, int lda, long long strideA, const float* B, int ldb,
                                         long long strideB, const float* beta, float* C, int ldc, long long strideC, int batch);
cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* alpha,
                            const void* A, cudaDataType at, int lda, const void* B, cudaDataType bt, int ldb, const void* beta,
                            void* C, cudaDataType ct, int ldc, cublasComputeType_t compute, cublasGemmAlgo_t algo);
