/*
 * Nested Array Optimizer for APL
 * High-performance C++ implementation of nested array operations
 * 
 * Strategy: Flatten nested arrays with offset-based indexing for cache efficiency
 * Supports SIMD vectorization and manual optimization opportunities
 */

#pragma once

#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <immintrin.h>  // SIMD intrinsics (AVX2/AVX512)
#include <cmath>

namespace APLOptimizer {

    /**
     * RaggedArray: Cache-friendly representation of nested arrays
     * 
     * Instead of:  [[1,2,3], [4,5], [6,7,8,9]]  (pointer-heavy, poor cache locality)
     * We store:    data=[1,2,3,4,5,6,7,8,9]
     *              offsets=[0, 3, 5, 9]
     * 
     * This converts random-access pointer chasing into two linear array accesses.
     */
    template<typename T>
    class RaggedArray {
    private:
        std::vector<T> data;          // Flattened array data
        std::vector<size_t> offsets;  // Start indices for each inner array
        
    public:
        RaggedArray() = default;
        
        /**
         * Build from nested array representation
         * Example: lengths = {3, 2, 4} means three inner arrays of those sizes
         */
        RaggedArray(const std::vector<std::vector<T>>& nested) {
            offsets.push_back(0);
            for (const auto& inner : nested) {
                data.insert(data.end(), inner.begin(), inner.end());
                offsets.push_back(data.size());
            }
        }
        
        /**
         * Get the i-th inner array (without copying - view semantics)
         * Returns pair of (start_ptr, length)
         */
        std::pair<T*, size_t> getInner(size_t i) const {
            if (i >= offsets.size() - 1) return {nullptr, 0};
            size_t start = offsets[i];
            size_t end = offsets[i + 1];
            return {data.data() + start, end - start};
        }
        
        /**
         * Direct access to inner array data
         */
        T* dataPtr() { return data.data(); }
        const T* dataPtr() const { return data.data(); }
        
        /**
         * Get offsets for bulk operations
         */
        const std::vector<size_t>& getOffsets() const { return offsets; }
        
        /**
         * Number of inner arrays (depth of nesting)
         */
        size_t numInnerArrays() const { return offsets.size() - 1; }
        
        /**
         * Total number of elements
         */
        size_t totalElements() const { return data.size(); }
        
        /**
         * Serialize to flat vector for transmission
         */
        std::vector<T> serialize() const { return data; }
    };
    
    
    // =====================================================================
    // SIMD Vectorized Operations
    // =====================================================================
    
    /**
     * SIMD Sum: Process 8 floats at once with AVX2
     * ~8x faster than scalar loop for large arrays
     */
    inline float simdSum(const std::vector<float>& arr) {
        if (arr.empty()) return 0.0f;
        
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        
        // Process 8 floats at a time
        for (; i + 8 <= arr.size(); i += 8) {
            __m256 chunk = _mm256_loadu_ps(&arr[i]);
            sum = _mm256_add_ps(sum, chunk);
        }
        
        // Horizontal sum: add all 8 floats together
        __m128 sum128 = _mm256_castps256_ps128(sum);
        sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(sum, 1));
        sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
        sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x33));
        float result = _mm_cvtss_f32(sum128);
        
        // Process remaining elements
        for (; i < arr.size(); i++) {
            result += arr[i];
        }
        
        return result;
    }
    
    /**
     * SIMD Element-wise Addition: arr1[i] + arr2[i]
     * ~8x faster for large arrays
     */
    inline std::vector<float> simdAdd(
        const std::vector<float>& arr1,
        const std::vector<float>& arr2
    ) {
        size_t minSize = std::min(arr1.size(), arr2.size());
        std::vector<float> result(minSize);
        
        size_t i = 0;
        
        // SIMD: 8 additions per iteration
        for (; i + 8 <= minSize; i += 8) {
            __m256 a = _mm256_loadu_ps(&arr1[i]);
            __m256 b = _mm256_loadu_ps(&arr2[i]);
            __m256 c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(&result[i], c);
        }
        
        // Scalar: remaining elements
        for (; i < minSize; i++) {
            result[i] = arr1[i] + arr2[i];
        }
        
        return result;
    }
    
    /**
     * SIMD Element-wise Multiplication: arr1[i] * arr2[i]
     */
    inline std::vector<float> simdMultiply(
        const std::vector<float>& arr1,
        const std::vector<float>& arr2
    ) {
        size_t minSize = std::min(arr1.size(), arr2.size());
        std::vector<float> result(minSize);
        
        size_t i = 0;
        
        for (; i + 8 <= minSize; i += 8) {
            __m256 a = _mm256_loadu_ps(&arr1[i]);
            __m256 b = _mm256_loadu_ps(&arr2[i]);
            __m256 c = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(&result[i], c);
        }
        
        for (; i < minSize; i++) {
            result[i] = arr1[i] * arr2[i];
        }
        
        return result;
    }
    
    
    // =====================================================================
    // Cache-Optimized Nested Array Operations
    // =====================================================================
    
    /**
     * Reduce operation across all inner arrays
     * Example: Sum each inner array, return vector of sums
     * 
     * Traditional APL: Requires pointer chasing for each inner array
     * This version: Linear scans with offset-based boundaries = cache-friendly
     */
    template<typename T, typename BinaryOp>
    std::vector<T> reduceInnerArrays(
        const RaggedArray<T>& ragged,
        T initial,
        BinaryOp op
    ) {
        std::vector<T> result;
        const auto& offsets = ragged.getOffsets();
        const T* data = ragged.dataPtr();
        
        // Process each inner array
        for (size_t i = 0; i < offsets.size() - 1; i++) {
            size_t start = offsets[i];
            size_t end = offsets[i + 1];
            
            T acc = initial;
            
            // Linear scan within this inner array - excellent cache locality
            for (size_t j = start; j < end; j++) {
                acc = op(acc, data[j]);
            }
            
            result.push_back(acc);
        }
        
        return result;
    }
    
    /**
     * Sum each inner array
     */
    template<typename T>
    std::vector<T> sumInnerArrays(const RaggedArray<T>& ragged) {
        return reduceInnerArrays(ragged, T(0), std::plus<T>());
    }
    
    /**
     * Find max in each inner array
     */
    template<typename T>
    std::vector<T> maxInnerArrays(const RaggedArray<T>& ragged) {
        std::vector<T> result;
        const auto& offsets = ragged.getOffsets();
        const T* data = ragged.dataPtr();
        
        for (size_t i = 0; i < offsets.size() - 1; i++) {
            size_t start = offsets[i];
            size_t end = offsets[i + 1];
            
            if (start >= end) continue;
            
            T maxVal = data[start];
            for (size_t j = start + 1; j < end; j++) {
                if (data[j] > maxVal) maxVal = data[j];
            }
            
            result.push_back(maxVal);
        }
        
        return result;
    }
    
    /**
     * Element-wise operation across all inner arrays
     * Example: Apply sin() to every element
     */
    template<typename T, typename UnaryOp>
    RaggedArray<T> mapElements(
        const RaggedArray<T>& ragged,
        UnaryOp op
    ) {
        std::vector<std::vector<T>> result_nested;
        const auto& offsets = ragged.getOffsets();
        const T* data = ragged.dataPtr();
        
        for (size_t i = 0; i < offsets.size() - 1; i++) {
            size_t start = offsets[i];
            size_t end = offsets[i + 1];
            
            std::vector<T> inner;
            for (size_t j = start; j < end; j++) {
                inner.push_back(op(data[j]));
            }
            
            result_nested.push_back(inner);
        }
        
        return RaggedArray<T>(result_nested);
    }
    
    /**
     * Grade (sort indices) within each inner array
     * Returns indices that would sort each inner array
     */
    template<typename T>
    std::vector<std::vector<size_t>> gradeInnerArrays(
        const RaggedArray<T>& ragged
    ) {
        std::vector<std::vector<size_t>> result;
        const auto& offsets = ragged.getOffsets();
        const T* data = ragged.dataPtr();
        
        for (size_t i = 0; i < offsets.size() - 1; i++) {
            size_t start = offsets[i];
            size_t end = offsets[i + 1];
            
            std::vector<size_t> indices(end - start);
            std::iota(indices.begin(), indices.end(), 0);
            
            // Sort indices by the values they point to
            std::sort(indices.begin(), indices.end(), 
                [data, start](size_t a, size_t b) {
                    return data[start + a] < data[start + b];
                });
            
            result.push_back(indices);
        }
        
        return result;
    }
    
    
    // =====================================================================
    // Performance Analysis
    // =====================================================================
    
    /**
     * Memory layout analysis for cache efficiency
     * Returns statistics about the ragged array structure
     */
    struct ArrayStatistics {
        size_t totalElements;
        size_t numInnerArrays;
        double avgInnerSize;
        double minInnerSize;
        double maxInnerSize;
        size_t totalMemoryBytes;
        
        void print() const {
            printf("=== Nested Array Statistics ===\n");
            printf("Total elements: %zu\n", totalElements);
            printf("Num inner arrays: %zu\n", numInnerArrays);
            printf("Avg inner array size: %.2f\n", avgInnerSize);
            printf("Min/Max inner array size: %zu / %zu\n", 
                   (size_t)minInnerSize, (size_t)maxInnerSize);
            printf("Total memory: %zu bytes (%.2f KB)\n", 
                   totalMemoryBytes, totalMemoryBytes / 1024.0);
            printf("Memory efficiency ratio: %.1f%%\n",
                   (100.0 * totalMemoryBytes) / (totalElements * sizeof(float) + numInnerArrays * 8));
        }
    };
    
    template<typename T>
    ArrayStatistics analyzeArray(const RaggedArray<T>& ragged) {
        const auto& offsets = ragged.getOffsets();
        ArrayStatistics stats{};
        
        stats.totalElements = ragged.totalElements();
        stats.numInnerArrays = ragged.numInnerArrays();
        
        if (stats.numInnerArrays > 0) {
            stats.avgInnerSize = (double)stats.totalElements / stats.numInnerArrays;
            
            size_t minSize = SIZE_MAX, maxSize = 0;
            for (size_t i = 0; i < offsets.size() - 1; i++) {
                size_t size = offsets[i + 1] - offsets[i];
                minSize = std::min(minSize, size);
                maxSize = std::max(maxSize, size);
            }
            
            stats.minInnerSize = minSize;
            stats.maxInnerSize = maxSize;
        }
        
        stats.totalMemoryBytes = stats.totalElements * sizeof(T) + 
                                 stats.numInnerArrays * sizeof(size_t);
        
        return stats;
    }

} // namespace APLOptimizer
