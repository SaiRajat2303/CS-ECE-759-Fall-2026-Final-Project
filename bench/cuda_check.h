#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                         __FILE__, __LINE__, cudaGetErrorString(err_));           \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
