#include "../include/quantized_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// --- NF4 Lookup Table (Constant Memory) ---
__constant__ float c_NF4_LUT[16] = {
    -1.00000000f, -0.69619280f, -0.52507305f, -0.39491749f,
    -0.28444138f, -0.18477343f, -0.09105004f,  0.00000000f,
     0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
     0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f
};

// Section 4.3: The Role of Assembly Language (PTX)
// This kernel uses inline PTX for the critical FMA instruction.
__global__ void QGEMM_CUDA_Kernel_PTX(float* C, const float* A, const Quantized_4Bit_Block* B_blocks, int M, int N, int K, QuantizationType quant_type) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Iterate over K in blocks
        int num_k_blocks = K / BLOCK_SIZE;
        for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
            // Locate the block for B(k, col)
            // Assuming B is stored column-major for efficiency in this kernel layout
            int block_idx = col * num_k_blocks + k_blk;
            
            Quantized_4Bit_Block q_block = B_blocks[block_idx];
            float scale = q_block.Scale_S;
            float zero_point = (float)q_block.ZeroPoint;
            
            // Process the block
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                // Dequantize weight
                uint8_t packed = q_block.Weights_Packed[i / 2];
                uint8_t nibble = (i % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                
                float weight;
                if (quant_type == QuantizationType::NF4_NORMAL_FLOAT) {
                     weight = scale * c_NF4_LUT[nibble];
                } else {
                     weight = scale * ((float)nibble - zero_point);
                }
                
                // Load A value
                int a_idx = row * K + (k_blk * BLOCK_SIZE + i);
                float val_a = A[a_idx];
                
                // --- INLINE PTX ASSEMBLY ---
                // Explicitly use the Fused Multiply-Add (FMA) instruction.
                // Syntax: fma.rn.f32 d, a, b, c;  // d = a*b + c
                // Constraints: "=f" (output float register), "f" (input float register)
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sum) : "f"(val_a), "f"(weight), "f"(sum));
            }
        }
        
        C[row * N + col] = sum;
    }
}

// Host wrapper function to manage memory and launch kernel
extern "C" bool Try_Launch_CUDA_QGEMM(float* C, const float* A, const Quantized_4Bit_Block* B_blocks_ptr, size_t num_blocks, int M, int N, int K, QuantizationType quant_type) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) return false;

    float *d_A, *d_C;
    Quantized_4Bit_Block *d_B_blocks;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    size_t size_B = num_blocks * sizeof(Quantized_4Bit_Block);
    
    // Allocate device memory
    if (cudaMalloc((void**)&d_A, size_A) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d_C, size_C) != cudaSuccess) { cudaFree(d_A); return false; }
    if (cudaMalloc((void**)&d_B_blocks, size_B) != cudaSuccess) { cudaFree(d_A); cudaFree(d_C); return false; }
    
    // Copy data to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_blocks, B_blocks_ptr, size_B, cudaMemcpyHostToDevice);
    
    // Launch Kernel
    // Using 16x16 threads per block
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    // Launch the PTX-optimized kernel
    QGEMM_CUDA_Kernel_PTX<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B_blocks, M, N, K, quant_type);
    
    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_C); cudaFree(d_B_blocks);
        return false;
    }

    // Copy result back
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_B_blocks);
    
    return true;
}
