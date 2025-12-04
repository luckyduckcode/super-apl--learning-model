// engine_wrapper.cpp
// Simplified C-callable wrapper for MatrixMultiply_Quantized
// Exposes a C interface that Python ctypes can call directly

#include "../include/quantized_types.h"
#include <cstring>
#include <cstdint>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

extern "C" {
    // Forward declare the main function (from qgemm_kernel.cpp)
    // Note: Signature updated to match qgemm_kernel.cpp
    int MatrixMultiply_Quantized(float* C, const float* A, const uint8_t* B_raw, int M, int N, int K);

    // Wrapper for the Quantized GEMM
    // Python should pass the raw bytes of the quantized blocks
    EXPORT int Run_Quantized_GEMM(float* C, const float* A, const uint8_t* B_quantized_bytes, int M, int N, int K) {
        return MatrixMultiply_Quantized(C, A, B_quantized_bytes, M, N, K);
    }

    // Legacy/Debug Wrapper: Standard GEMM
    EXPORT void SimpleMatrixMultiply(float* C, const float* A, const float* W, int M, int N, int K) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[m * K + k] * W[k * N + n];
                }
                C[m * N + n] = sum;
            }
        }
    }
}
