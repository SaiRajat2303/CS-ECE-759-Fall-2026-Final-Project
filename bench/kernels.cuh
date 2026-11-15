#pragma once
#include "lbvh/kernels_impl.h"

__global__ void morton_kernel(const lbvh::Vec3* __restrict__ points,
                              const lbvh::AABB* __restrict__ scene, int n,
                              uint32_t* __restrict__ keys, uint32_t* __restrict__ vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lbvh::mortonBody(i, points, scene, keys, vals);
}

__global__ void hierarchy_kernel(const uint32_t* __restrict__ keys, int n,
                                 lbvh::NodeView nodes, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) lbvh::hierarchyBody(i, keys, n, nodes, flags);
}

__global__ void refit_kernel(int n, const lbvh::Vec3* __restrict__ points,
                             const uint32_t* __restrict__ sorted_prims,
                             lbvh::NodeView nodes, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lbvh::refitBody(i, n, points, sorted_prims, nodes, flags);
}
