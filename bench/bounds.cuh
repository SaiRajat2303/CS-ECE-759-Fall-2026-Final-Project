#pragma once
#include "lbvh/common.h"
#include "prims.h"
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <cstdio>
#include <cstdlib>

struct PointToBox {
    __host__ __device__ lbvh::AABB operator()(const lbvh::Vec3& p) const { return lbvh::pointBox(p); }
};

struct CentroidToBox {
    __host__ __device__ lbvh::AABB operator()(const lbvh::AABB& b) const { return lbvh::pointBox(box_centroid(b)); }
};

/* min and max are exact and order-independent, so the reduced box is bit-identical to a serial CPU merge. */
struct BoxMerge {
    __host__ __device__ lbvh::AABB operator()(const lbvh::AABB& a, const lbvh::AABB& b) const {
        return lbvh::merge(a, b);
    }
};

inline auto point_boxes(const lbvh::Vec3* p) { return thrust::make_transform_iterator(p, PointToBox{}); }
inline auto centroid_boxes(const lbvh::AABB* b) { return thrust::make_transform_iterator(b, CentroidToBox{}); }

template <typename It>
inline size_t bounds_temp_bytes(It in, int n) {
    size_t bytes = 0;
    cudaError_t e = cub::DeviceReduce::Reduce(nullptr, bytes, in, (lbvh::AABB*)nullptr, n, BoxMerge{}, lbvh::emptyBox());
    if (e != cudaSuccess || bytes == 0) {
        std::fprintf(stderr, "bounds_temp_bytes failed: %s (bytes=%zu)\n", cudaGetErrorString(e), bytes);
        std::exit(1);
    }
    return bytes;
}

template <typename It>
inline cudaError_t reduce_bounds(void* temp, size_t bytes, It in, lbvh::AABB* d_out, int n, cudaStream_t stream) {
    return cub::DeviceReduce::Reduce(temp, bytes, in, d_out, n, BoxMerge{}, lbvh::emptyBox(), stream);
}
