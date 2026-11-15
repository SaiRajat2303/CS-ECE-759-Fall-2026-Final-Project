#pragma once
#include "lbvh/common.h"
#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

inline size_t sort_temp_bytes(int n) {
    size_t bytes = 0;
    cudaError_t e = cub::DeviceRadixSort::SortPairs(nullptr, bytes,
                                                    (const uint32_t*)nullptr, (uint32_t*)nullptr,
                                                    (const uint32_t*)nullptr, (uint32_t*)nullptr,
                                                    n, 0, lbvh::kMortonBits);
    if (e != cudaSuccess || bytes == 0) {
        std::fprintf(stderr, "sort_temp_bytes failed: %s (bytes=%zu)\n", cudaGetErrorString(e), bytes);
        std::exit(1);
    }
    return bytes;
}

/* LSD radix sort is stable, so equal Morton codes keep ascending input order, as the CPU reference expects. */
inline cudaError_t sort_codes(void* temp, size_t bytes,
                              const uint32_t* d_keys_in, uint32_t* d_keys_out,
                              const uint32_t* d_vals_in, uint32_t* d_vals_out,
                              int n, cudaStream_t stream) {
    return cub::DeviceRadixSort::SortPairs(temp, bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out,
                                           n, 0, lbvh::kMortonBits, stream);
}
