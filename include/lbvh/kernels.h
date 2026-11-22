// kernels.h -- declarations of the GPU kernels defined in src/kernels.cu.
//
// kernels.cu is compiled on its own so that the register cap (-maxrregcount)
// applies to these kernels only and not to CUB's sort and reduce kernels.
#pragma once

#include <cuda_runtime.h>

#include "lbvh/common.h"
#include "lbvh/nodes.h"

namespace lbvh {

// Stage 2. n threads: Morton code and index of point i.
__global__ void mortonKernel(const Vec3* points, const AABB* sceneBox, uint32_t* keys,
                             uint32_t* vals, int n);

// Stage 4. n-1 threads: links of internal node i, from the sorted keys.
__global__ void hierarchyKernel(const uint32_t* sortedKeys, int n, NodeView nodes, int* flags);

// Stage 5. n threads: write leaf i, then merge boxes up towards the root.
__global__ void refitKernel(const Vec3* points, const uint32_t* sortedPrims, int n, NodeView nodes,
                            int* flags);

// Fixed query kernel (not tuned). numQueries threads: number of points
// within the radius of query i.
__global__ void radiusCountKernel(NodeView nodes, const Vec3* queries, int numQueries,
                                  float radiusSquared, int* counts);

constexpr int kQueryBlockSize = 256;

// Link-time guard: the symbol name depends on the node layout, so objects
// compiled with different layouts cannot be linked together by mistake.
#if LBVH_LAYOUT_SOA
#define LBVH_ABI_TAG nodeLayoutIsSoA
#else
#define LBVH_ABI_TAG nodeLayoutIsAoS
#endif
extern const int LBVH_ABI_TAG;

}  // namespace lbvh
