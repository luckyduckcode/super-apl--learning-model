#include "quantized_types.h"
#include <immintrin.h>
#include <iostream>

#ifdef USE_CUDA
// Forward declaration of the CUDA launcher
extern "C" bool Try_Launch_CUDA_QGEMM(float* C, const float* A, const Quantized_4Bit_Block* B_blocks_ptr, size_t num_blocks, int M, int N, int K, QuantizationType quant_type);
#endif

// Helper to unpack 4-bit nibbles to 32-bit floats (simplified for demonstration)
// In a real high-perf scenario, we'd use _mm256_shuffle_epi8 and integer arithmetic before converting to float.
// Here we follow the paper's logic conceptually.

// Section 3.1: SIMD_Q_GEMM_Kernel
// Computes dot product of a float row and a quantized column block
float SIMD_Q_GEMM_Kernel(const float* A_row, const Quantized_4Bit_Block& B_col_quant, QuantizationType quant_type) {
    
    __m256 sum_vec = _mm256_setzero_ps();
    
    // Load scale
    __m256 scale_vec = _mm256_set1_ps(B_col_quant.Scale_S);
    
    // Zero point is only used for INT4_LINEAR
    __m256 zero_point_vec = _mm256_setzero_ps();
    if (quant_type == QuantizationType::INT4_LINEAR) {
        zero_point_vec = _mm256_set1_ps((float)B_col_quant.ZeroPoint);
    }

    // Loop over the inner dimension (K) in chunks compatible with SIMD width
    // We process 8 elements at a time (AVX2 is 256-bit, holding 8 floats)
    for (int i = 0; i < BLOCK_SIZE; i += 8) {
        // --- STEP 1: LOAD and UNPACK/DEQUANTIZE 4-bit Weights ---
        
        // Load 4 bytes (containing 8 packed 4-bit weights)
        // Note: This is a simplified load. In reality, we need to handle alignment carefully.
        // We interpret the packed weights as 32-bit int to load 4 bytes easily, or just load byte by byte.
        
        // We need to unpack 4 bytes into 8 floats.
        float unpacked_weights[8];
        for (int j = 0; j < 4; ++j) {
            uint8_t packed = B_col_quant.Weights_Packed[(i/2) + j];
            uint8_t low = packed & 0x0F;
            uint8_t high = (packed >> 4) & 0x0F;
            
            if (quant_type == QuantizationType::NF4_NORMAL_FLOAT) {
                // Use Lookup Table for NF4
                unpacked_weights[2*j] = NF4_LUT[low];
                unpacked_weights[2*j+1] = NF4_LUT[high];
            } else {
                // Standard INT4
                unpacked_weights[2*j] = (float)low;
                unpacked_weights[2*j+1] = (float)high;
            }
        }

        __m256 q_vec = _mm256_loadu_ps(unpacked_weights);
        __m256 weight_fp32;

        if (quant_type == QuantizationType::NF4_NORMAL_FLOAT) {
             // NF4: r = S * LUT[q]
             weight_fp32 = _mm256_mul_ps(q_vec, scale_vec);
        } else {
             // INT4: r = S * (q - Z)
             __m256 diff_vec = _mm256_sub_ps(q_vec, zero_point_vec);
             weight_fp32 = _mm256_mul_ps(diff_vec, scale_vec);
        }

        // --- STEP 2: LOAD and PARALLEL MULTIPLY (FMA) ---
        __m256 activation_chunk = _mm256_loadu_ps(A_row + i);
        
        // Fused Multiply-Add (FMA) Instruction: Accumulator_Vector = A * B + Accumulator_Vector
        sum_vec = _mm256_fmadd_ps(activation_chunk, weight_fp32, sum_vec);
    }

    // --- STEP 3: REDUCE and STORE ---
    // Horizontal sum of the vector
    float result_arr[8];
    _mm256_storeu_ps(result_arr, sum_vec);
    
    float total = 0.0f;
    for (int k = 0; k < 8; ++k) {
        total += result_arr[k];
    }
    
    return total;
}

// The Engine Dispatch function (Section 1.1 Middle Layer)
extern "C" {
    // This function acts as the "Middle Layer" described in the architecture.
    // It receives requests from the Top Layer (APL) and dispatches them to the 
    // Bottom Layer (Optimized Kernels: AVX2, CUDA/PTX, or Tensor Cores).
    //
    // C is M x N, A is M x K, B is K x N (quantized)
    // Returns 0 on success.
    int MatrixMultiply_Quantized(float* C, const float* A, const uint8_t* B_raw, int M, int N, int K) {
        
        // Cast raw bytes to Quantized_4Bit_Block
        // We assume B_raw contains a flat array of packed blocks.
        const Quantized_4Bit_Block* blocks = reinterpret_cast<const Quantized_4Bit_Block*>(B_raw);
        
        // Default to NF4 as per research paper "idea"
        QuantizationType q_type = QuantizationType::NF4_NORMAL_FLOAT;

        #ifdef USE_CUDA
        // Check for FP8 usage (Section 5.2)
        // Note: FP8 check removed from here as we don't have the struct to check 'use_fp8'.
        // We assume NF4 for this raw interface.

        // Attempt to run on GPU first using the PTX-optimized kernel
        // Calculate total blocks: (K / BLOCK_SIZE) * N
        size_t total_blocks = (size_t)(K / BLOCK_SIZE) * N;
        if (Try_Launch_CUDA_QGEMM(C, A, blocks, total_blocks, M, N, K, q_type)) {
            return 0;
        }
        #endif

        // Fallback to CPU (AVX2 SIMD)
        // Naive loop structure for demonstration. 
        // Real implementation would tile this for cache efficiency.
        
        // #pragma omp parallel for collapse(2) // Uncomment if OpenMP is configured
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                
                // Iterate over blocks in the K dimension
                // Assuming K is a multiple of BLOCK_SIZE for simplicity
                for (int k_blk = 0; k_blk < K / BLOCK_SIZE; ++k_blk) {
                    // Locate the specific block in the quantized matrix B
                    // This indexing depends on how B is flattened. 
                    // Let's assume B is column-major or block-major for efficiency.
                    // Index = n * (K / BLOCK_SIZE) + k_blk
                    int block_idx = n * (K / BLOCK_SIZE) + k_blk;
                    
                    const Quantized_4Bit_Block& block = blocks[block_idx];
                    
                    // Pointer to the start of the current segment of A's row
                    const float* A_ptr = A + (m * K) + (k_blk * BLOCK_SIZE);
                    
                    sum += SIMD_Q_GEMM_Kernel(A_ptr, block, q_type);
                }
                
                C[m * N + n] = sum;
            }
        }
        return 0;
    }
}
