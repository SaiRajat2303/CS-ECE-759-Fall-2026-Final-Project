#pragma once
#include "lbvh/kernels_impl.h"
#include "prims.h"

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

/* Triangle input: the Morton code comes from the box centroid, quantized against the centroid bounds. */
__global__ void morton_box_kernel(const lbvh::AABB* __restrict__ boxes,
                                  const lbvh::AABB* __restrict__ centroid_bounds, int n,
                                  uint32_t* __restrict__ keys, uint32_t* __restrict__ vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = lbvh::mortonCode(box_centroid(boxes[i]), *centroid_bounds);
    vals[i] = (uint32_t)i;
}

/* Same walk as lbvh::refitBody, except a leaf takes its triangle's box instead of a point box. */
__global__ void refit_box_kernel(int n, const lbvh::AABB* __restrict__ boxes,
                                 const uint32_t* __restrict__ sorted_prims,
                                 lbvh::NodeView nodes, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int prim = (int)lbvh::loadStream(sorted_prims + i);
    int cur = lbvh::leafIndex(n, i);
    lbvh::AABB box = boxes[prim];
    lbvh::node::setChildren(nodes, cur, -1, -1);
    lbvh::node::setPrim(nodes, cur, prim);
    lbvh::node::setBox(nodes, cur, box);

    int parent = lbvh::node::parent(nodes, cur);
    while (parent >= 0) {
        lbvh::publishStores();
        if (lbvh::atomicFetchInc(&flags[parent]) == 0) return;
        const int l = lbvh::node::left(nodes, parent);
        const int r = lbvh::node::right(nodes, parent);
        box = lbvh::merge(box, lbvh::node::boxShared(nodes, l == cur ? r : l));
        lbvh::node::setBox(nodes, parent, box);
        cur = parent;
        parent = lbvh::node::parent(nodes, cur);
    }
}
