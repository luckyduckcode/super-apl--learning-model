#pragma once
#include <cstdint>
#include <vector>

// Block size for quantization (e.g., 32 or 64)
constexpr int BLOCK_SIZE = 32;

enum class QuantizationType {
    INT4_LINEAR,
    NF4_NORMAL_FLOAT
};

// NF4 Lookup Table (Standard Normal Float 4 values)
// These values are derived from the normal distribution.
static const float NF4_LUT[16] = {
    -1.00000000f, -0.69619280f, -0.52507305f, -0.39491749f,
    -0.28444138f, -0.18477343f, -0.09105004f,  0.00000000f,
     0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
     0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f
};

// Section 2.1: Quantized Weight Structure
// For 4-bit quantization, weights are stored in a packed format.
#pragma pack(push, 1)
struct Quantized_4Bit_Block {
    float Scale_S;      // Scale factor for the block
    uint8_t ZeroPoint;  // Zero-point for the block (Used for INT4, ignored for NF4)
    uint8_t Weights_Packed[BLOCK_SIZE / 2]; // Packed 4-bit integers
};
#pragma pack(pop)

// Structure to represent a full matrix of quantized weights
struct QuantizedMatrix {
    int rows;
    int cols;
    QuantizationType quant_type; // Added to support NF4 vs INT4
    bool use_fp8;                // Added to support FP8 check
    std::vector<Quantized_4Bit_Block> blocks;
};
