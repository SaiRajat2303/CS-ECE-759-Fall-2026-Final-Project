// lbvh_user_bvh.cuh -- the LBVH from include/lbvh/ plugged into the dragon
// benchmark's UserBVH adapter.
//
// Include this from bvh_cuda_bench.cu AFTER Ray, Hit and CUDA_CHECK are
// defined. Link with src/kernels.cu (Morton and hierarchy kernels).
//
// One leaf per triangle. Pipeline per build (all on the caller's stream):
//   0. triangle boxes + centroids      triBoundsKernel        (tri_impl.h)
//   1. scene box of the centroids      cub::DeviceReduce
//   2. Morton codes of the centroids   lbvh::mortonKernel     (kernels.cu)
//   3. radix sort                      cub::DeviceRadixSort
//   4. hierarchy                       lbvh::hierarchyKernel  (kernels.cu)
//   5. refit with triangle boxes       refitTriKernel         (tri_impl.h)
// update() keeps the topology and only redoes stages 0 and 5 (refit).
#pragma once

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "lbvh/kernels.h"
#include "lbvh/tri_impl.h"

namespace tribvh {

using lbvh::AABB;
using lbvh::NodeView;
using lbvh::Tri;
using lbvh::Vec3;

// float3 / int3 and Vec3 / Tri have the same size and alignment, so the
// harness's buffers are used in place.
static_assert(sizeof(float3) == sizeof(Vec3) && alignof(float3) == alignof(Vec3), "float3 != Vec3");
static_assert(sizeof(int3) == sizeof(Tri) && alignof(int3) == alignof(Tri), "int3 != Tri");

inline int gridFor(int count, int block) { return (count + block - 1) / block; }

struct PointToBox {
  __host__ __device__ AABB operator()(const Vec3& p) const { return lbvh::pointBox(p); }
};
struct MergeBoxes {
  __host__ __device__ AABB operator()(const AABB& a, const AABB& b) const { return lbvh::merge(a, b); }
};
struct SahOfNode {
  NodeView nodes;
  __host__ __device__ double operator()(int k) const { return lbvh::sahNodeCost(nodes, k); }
};

constexpr int kTraceBlock = 128;

__global__ void __launch_bounds__(LBVH_BLOCK_SIZE)
triBoundsKernel(const Vec3* verts, const Tri* tris, AABB* boxes, Vec3* centroids, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) lbvh::triBoundsBody(i, verts, tris, boxes, centroids);
}

__global__ void __launch_bounds__(LBVH_BLOCK_SIZE)
refitTriKernel(const AABB* triBoxes, const uint32_t* sortedPrims, int n, NodeView nodes, int* flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) lbvh::refitTriBody(i, n, triBoxes, sortedPrims, nodes, flags);
}

__device__ inline lbvh::RayQuery toQuery(const Ray& r) {
  return lbvh::RayQuery{{r.o.x, r.o.y, r.o.z}, {r.d.x, r.d.y, r.d.z}, r.tmin, r.tmax};
}

__global__ void __launch_bounds__(kTraceBlock)
traceKernel(NodeView nodes, int n, const Vec3* verts, const Tri* tris, const Ray* rays, Hit* hits,
            int numRays) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numRays) return;
  const lbvh::TriHit h = lbvh::traceClosestBody(nodes, n, verts, tris, toQuery(rays[i]));
  hits[i] = Hit{h.t, h.prim, h.u, h.v};
}

// Instrumented traversal: same walk, only the number of box tests is kept.
__global__ void __launch_bounds__(kTraceBlock)
traceVisitsKernel(NodeView nodes, int n, const Vec3* verts, const Tri* tris, const Ray* rays,
                  int numRays, unsigned long long* totalVisits) {
  using BlockReduce = cub::BlockReduce<unsigned long long, kTraceBlock>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int visits = 0;
  if (i < numRays) lbvh::traceClosestBody(nodes, n, verts, tris, toQuery(rays[i]), &visits);
  const unsigned long long sum = BlockReduce(temp).Sum(static_cast<unsigned long long>(visits));
  if (threadIdx.x == 0) atomicAdd(totalVisits, sum);
}

}  // namespace tribvh

struct UserBVH {
  int n = 0;
  lbvh::AABB* dTriBoxes = nullptr;
  lbvh::Vec3* dCentroids = nullptr;
  lbvh::AABB* dSceneBox = nullptr;
  uint32_t* dKeys[2] = {nullptr, nullptr};
  uint32_t* dVals[2] = {nullptr, nullptr};
  const uint32_t* dSortedPrims = nullptr;  // points into dVals[0 or 1] after build()
  void* dNodeStorage = nullptr;
  lbvh::NodeView nodes{};
  int* dFlags = nullptr;
  void* dReduceTemp = nullptr;
  size_t reduceTempBytes = 0;
  void* dSortTemp = nullptr;
  size_t sortTempBytes = 0;
  void* dSahTemp = nullptr;
  size_t sahTempBytes = 0;
  double* dSah = nullptr;
  unsigned long long* dVisits = nullptr;
  bool built = false;

  int numNodes() const { return 2 * n - 1; }

  void init(int numVerts, int numTris) {
    (void)numVerts;
    if (lbvh::LBVH_ABI_TAG != 1) std::abort();  // kernels.cu built with another node layout
    if (numTris < 1) {
      std::fprintf(stderr, "UserBVH: mesh has no triangles\n");
      std::exit(1);
    }
    n = numTris;
    const size_t nn = static_cast<size_t>(numNodes());
    CUDA_CHECK(cudaMalloc(&dTriBoxes, sizeof(lbvh::AABB) * n));
    CUDA_CHECK(cudaMalloc(&dCentroids, sizeof(lbvh::Vec3) * n));
    CUDA_CHECK(cudaMalloc(&dSceneBox, sizeof(lbvh::AABB)));
    for (int k = 0; k < 2; ++k) {
      CUDA_CHECK(cudaMalloc(&dKeys[k], sizeof(uint32_t) * n));
      CUDA_CHECK(cudaMalloc(&dVals[k], sizeof(uint32_t) * n));
    }
    CUDA_CHECK(cudaMalloc(&dNodeStorage, lbvh::nodeStorageBytes(nn)));
    nodes = lbvh::makeNodeView(dNodeStorage, nn);
    CUDA_CHECK(cudaMalloc(&dFlags, sizeof(int) * (n > 1 ? n - 1 : 1)));
    CUDA_CHECK(cudaMalloc(&dSah, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dVisits, sizeof(unsigned long long)));

    // CUB scratch sizes, so nothing is allocated inside build()/update().
    auto boxes = thrust::make_transform_iterator(dCentroids, tribvh::PointToBox());
    CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, reduceTempBytes, boxes, dSceneBox, n,
                                         tribvh::MergeBoxes(), lbvh::emptyBox()));
    CUDA_CHECK(cudaMalloc(&dReduceTemp, reduceTempBytes ? reduceTempBytes : 1));

    cub::DoubleBuffer<uint32_t> keys(dKeys[0], dKeys[1]), vals(dVals[0], dVals[1]);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempBytes, keys, vals, n, 0,
                                               lbvh::kMortonBits));
    CUDA_CHECK(cudaMalloc(&dSortTemp, sortTempBytes ? sortTempBytes : 1));

    auto costs = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                 tribvh::SahOfNode{nodes});
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, sahTempBytes, costs, dSah, numNodes()));
    CUDA_CHECK(cudaMalloc(&dSahTemp, sahTempBytes ? sahTempBytes : 1));
  }

  // Full build from scratch: stages 0-5.
  void build(const float3* dVerts, const int3* dTris, int numVerts, int numTris, cudaStream_t s) {
    (void)numVerts; (void)numTris;
    const int block = LBVH_BLOCK_SIZE;
    const auto* verts = reinterpret_cast<const lbvh::Vec3*>(dVerts);
    const auto* tris = reinterpret_cast<const lbvh::Tri*>(dTris);

    tribvh::triBoundsKernel<<<tribvh::gridFor(n, block), block, 0, s>>>(verts, tris, dTriBoxes,
                                                                         dCentroids, n);
    auto boxes = thrust::make_transform_iterator(dCentroids, tribvh::PointToBox());
    size_t reduceBytes = reduceTempBytes;
    CUDA_CHECK(cub::DeviceReduce::Reduce(dReduceTemp, reduceBytes, boxes, dSceneBox, n,
                                         tribvh::MergeBoxes(), lbvh::emptyBox(), s));
    lbvh::mortonKernel<<<tribvh::gridFor(n, block), block, 0, s>>>(dCentroids, dSceneBox,
                                                                    dKeys[0], dVals[0], n);
    cub::DoubleBuffer<uint32_t> keys(dKeys[0], dKeys[1]), vals(dVals[0], dVals[1]);
    size_t sortBytes = sortTempBytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dSortTemp, sortBytes, keys, vals, n, 0,
                                               lbvh::kMortonBits, s));
    dSortedPrims = vals.Current();
    if (n > 1) {
      lbvh::hierarchyKernel<<<tribvh::gridFor(n - 1, block), block, 0, s>>>(keys.Current(), n,
                                                                              nodes, dFlags);
    } else {
      // A single triangle: the root is the only leaf and has no parent.
      const int noParent = -1;
      CUDA_CHECK(cudaMemcpyAsync(&LBVH_FIELD(nodes, parent, 0), &noParent, sizeof(int),
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaStreamSynchronize(s));  // noParent lives on this stack frame
    }
    tribvh::refitTriKernel<<<tribvh::gridFor(n, block), block, 0, s>>>(dTriBoxes, dSortedPrims, n,
                                                                        nodes, dFlags);
    CUDA_CHECK(cudaGetLastError());
    built = true;
  }

  // Dynamic update: keep the Morton order and topology from the last build,
  // recompute triangle boxes for the new vertices and refit bottom-up.
  void update(const float3* dVerts, const int3* dTris, int numVerts, int numTris, cudaStream_t s) {
    if (!built) { build(dVerts, dTris, numVerts, numTris, s); return; }
    const int block = LBVH_BLOCK_SIZE;
    tribvh::triBoundsKernel<<<tribvh::gridFor(n, block), block, 0, s>>>(
        reinterpret_cast<const lbvh::Vec3*>(dVerts), reinterpret_cast<const lbvh::Tri*>(dTris),
        dTriBoxes, dCentroids, n);
    if (n > 1) CUDA_CHECK(cudaMemsetAsync(dFlags, 0, sizeof(int) * (n - 1), s));
    tribvh::refitTriKernel<<<tribvh::gridFor(n, block), block, 0, s>>>(dTriBoxes, dSortedPrims, n,
                                                                        nodes, dFlags);
    CUDA_CHECK(cudaGetLastError());
  }

  // Closest-hit traversal.
  void trace(const Ray* dRays, Hit* dHits, int numRays, const float3* dVerts, const int3* dTris,
             cudaStream_t s) {
    tribvh::traceKernel<<<tribvh::gridFor(numRays, tribvh::kTraceBlock), tribvh::kTraceBlock, 0, s>>>(
        nodes, n, reinterpret_cast<const lbvh::Vec3*>(dVerts),
        reinterpret_cast<const lbvh::Tri*>(dTris), dRays, dHits, numRays);
  }

  // SAH cost normalised by the root's surface area (C_trav = 1.2, C_isect = 1).
  double sahCost(cudaStream_t s) {
    auto costs = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                 tribvh::SahOfNode{nodes});
    size_t bytes = sahTempBytes;
    CUDA_CHECK(cub::DeviceReduce::Sum(dSahTemp, bytes, costs, dSah, numNodes(), s));
    double total = 0;
    lbvh::AABB root{};
    CUDA_CHECK(cudaMemcpyAsync(&total, dSah, sizeof(double), cudaMemcpyDeviceToHost, s));
#if LBVH_LAYOUT_SOA
    float* fields[6] = {nodes.lox, nodes.loy, nodes.loz, nodes.hix, nodes.hiy, nodes.hiz};
    float* dst[6] = {&root.lo.x, &root.lo.y, &root.lo.z, &root.hi.x, &root.hi.y, &root.hi.z};
    for (int f = 0; f < 6; ++f)
      CUDA_CHECK(cudaMemcpyAsync(dst[f], fields[f], sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
#else
    lbvh::NodeAoS r{};
    CUDA_CHECK(cudaMemcpyAsync(&r, nodes.n, sizeof r, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    root = lbvh::AABB{{r.lox, r.loy, r.loz}, {r.hix, r.hiy, r.hiz}};
#endif
    const double rootArea = lbvh::surfaceArea(root);
    return rootArea > 0 ? total / rootArea : -1.0;
  }

  double visitsPerRay(const Ray* dRays, int numRays, const float3* dVerts, const int3* dTris,
                      cudaStream_t s) {
    if (numRays <= 0) return -1.0;
    CUDA_CHECK(cudaMemsetAsync(dVisits, 0, sizeof(unsigned long long), s));
    tribvh::traceVisitsKernel<<<tribvh::gridFor(numRays, tribvh::kTraceBlock), tribvh::kTraceBlock, 0, s>>>(
        nodes, n, reinterpret_cast<const lbvh::Vec3*>(dVerts),
        reinterpret_cast<const lbvh::Tri*>(dTris), dRays, numRays, dVisits);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&total, dVisits, sizeof total, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return static_cast<double>(total) / numRays;
  }

  void destroy() {
    cudaFree(dTriBoxes); cudaFree(dCentroids); cudaFree(dSceneBox);
    for (int k = 0; k < 2; ++k) { cudaFree(dKeys[k]); cudaFree(dVals[k]); }
    cudaFree(dNodeStorage); cudaFree(dFlags);
    cudaFree(dReduceTemp); cudaFree(dSortTemp); cudaFree(dSahTemp);
    cudaFree(dSah); cudaFree(dVisits);
    *this = UserBVH{};
  }
};
