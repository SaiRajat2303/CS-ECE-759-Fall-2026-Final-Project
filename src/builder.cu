// builder.cu -- buffers, host-memory modes, kernel launches and CUDA Graph.
//
// Compiled without -maxrregcount so CUB's kernels keep their own tuning.
#include "lbvh/builder.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

#include "lbvh/export.h"
#include "lbvh/kernels.h"

namespace lbvh {

namespace {

#define LBVH_CUDA_CHECK(call)                                                          \
  do {                                                                                 \
    const cudaError_t lbvhErr = (call);                                                \
    if (lbvhErr != cudaSuccess) {                                                      \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(lbvhErr),  \
                   __FILE__, __LINE__, cudaGetErrorString(lbvhErr));                   \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Functors for the scene-box reduction: each point becomes a zero-size box,
// and boxes are merged pairwise.
struct PointToBox {
  LBVH_HD AABB operator()(const Vec3& p) const { return pointBox(p); }
};
struct MergeBoxes {
  LBVH_HD AABB operator()(const AABB& a, const AABB& b) const { return merge(a, b); }
};

int gridSize(int count, int blockSize) { return (count + blockSize - 1) / blockSize; }

template <typename T>
void deviceAlloc(T*& ptr, size_t bytes) {
  void* raw = nullptr;
  LBVH_CUDA_CHECK(cudaMalloc(&raw, bytes ? bytes : 1));
  ptr = static_cast<T*>(raw);
}

}  // namespace

const char* hostMemName(HostMem m) {
  switch (m) {
    case HostMem::Pageable: return "pageable";
    case HostMem::Pinned: return "pinned";
    case HostMem::Unified: return "unified";
    case HostMem::UnifiedPrefetch: return "unified_prefetch";
  }
  return "?";
}

const char* launchModeName(LaunchMode m) { return m == LaunchMode::Graph ? "graph" : "separate"; }

bool parseHostMem(const std::string& s, HostMem& m) {
  if (s == "pageable") m = HostMem::Pageable;
  else if (s == "pinned") m = HostMem::Pinned;
  else if (s == "unified") m = HostMem::Unified;
  else if (s == "unified_prefetch") m = HostMem::UnifiedPrefetch;
  else return false;
  return true;
}

bool parseLaunchMode(const std::string& s, LaunchMode& m) {
  if (s == "separate") m = LaunchMode::Separate;
  else if (s == "graph") m = LaunchMode::Graph;
  else return false;
  return true;
}

// ---------------------------------------------------------------------------
// Construction: allocate everything once, outside the timed build
// ---------------------------------------------------------------------------

Builder::Builder(int n, HostMem hostMem, LaunchMode launch)
    : n_(n), hostMem_(hostMem), launch_(launch) {
  if (n < 2) {
    std::fprintf(stderr, "lbvh::Builder needs at least 2 points (got %d)\n", n);
    std::exit(1);
  }
  // Referencing the tag makes the link fail if kernels.cu was compiled with a
  // different node layout than this file.
  if (LBVH_ABI_TAG != 1) std::abort();

  LBVH_CUDA_CHECK(cudaGetDevice(&device_));
  LBVH_CUDA_CHECK(cudaStreamCreate(&stream_));
  for (auto& e : events_) LBVH_CUDA_CHECK(cudaEventCreate(&e));

  const size_t pointBytes = static_cast<size_t>(n_) * sizeof(Vec3);
  const size_t wordBytes = static_cast<size_t>(n_) * sizeof(uint32_t);
  const size_t numNodes = 2 * static_cast<size_t>(n_) - 1;

  // Host-side input buffer, and where the kernels read the points from.
  switch (hostMem_) {
    case HostMem::Pageable:
      hostPoints_ = static_cast<Vec3*>(std::malloc(pointBytes));
      if (!hostPoints_) {
        std::fprintf(stderr, "out of host memory\n");
        std::exit(1);
      }
      deviceAlloc(devicePoints_, pointBytes);
      break;
    case HostMem::Pinned:
      {
        void* raw = nullptr;
        LBVH_CUDA_CHECK(cudaHostAlloc(&raw, pointBytes, cudaHostAllocDefault));
        hostPoints_ = static_cast<Vec3*>(raw);
      }
      deviceAlloc(devicePoints_, pointBytes);
      break;
    case HostMem::Unified:
    case HostMem::UnifiedPrefetch:
    {
      void* raw = nullptr;
      LBVH_CUDA_CHECK(cudaMallocManaged(&raw, pointBytes, cudaMemAttachGlobal));
      hostPoints_ = static_cast<Vec3*>(raw);
      devicePoints_ = hostPoints_;  // one buffer, visible to both sides
    }
      break;
  }

  deviceAlloc(sceneBox_, sizeof(AABB));
  for (int k = 0; k < 2; ++k) {
    deviceAlloc(keys_[k], wordBytes);
    deviceAlloc(vals_[k], wordBytes);
  }
  deviceAlloc(nodeStorage_, nodeStorageBytes(numNodes));
  nodes_ = makeNodeView(nodeStorage_, numNodes);
  deviceAlloc(flags_, static_cast<size_t>(n_ - 1) * sizeof(int));

  // CUB needs scratch memory. Ask each algorithm how much (first call, with a
  // null pointer) and allocate it here, so no allocation happens in the build.
  {
    auto boxes = thrust::make_transform_iterator(devicePoints_, PointToBox());
    LBVH_CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, reduceTempBytes_, boxes, sceneBox_, n_,
                                              MergeBoxes(), emptyBox(), stream_));
    deviceAlloc(reduceTemp_, reduceTempBytes_);

    cub::DoubleBuffer<uint32_t> keys(keys_[0], keys_[1]);
    cub::DoubleBuffer<uint32_t> vals(vals_[0], vals_[1]);
    LBVH_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortTempBytes_, keys, vals, n_, 0,
                                                    kMortonBits, stream_));
    deviceAlloc(sortTemp_, sortTempBytes_);
  }

  deviceBytes_ = sizeof(AABB) + 4 * wordBytes + nodeStorageBytes(numNodes) +
                 static_cast<size_t>(n_ - 1) * sizeof(int) + reduceTempBytes_ + sortTempBytes_ +
                 pointBytes;

  if (launch_ == LaunchMode::Graph) captureGraph();
}

Builder::~Builder() {
  cudaStreamSynchronize(stream_);
  if (graphExec_) cudaGraphExecDestroy(graphExec_);
  if (graph_) cudaGraphDestroy(graph_);
  switch (hostMem_) {
    case HostMem::Pageable:
      std::free(hostPoints_);
      cudaFree(devicePoints_);
      break;
    case HostMem::Pinned:
      cudaFreeHost(hostPoints_);
      cudaFree(devicePoints_);
      break;
    case HostMem::Unified:
    case HostMem::UnifiedPrefetch:
      cudaFree(hostPoints_);
      break;
  }
  cudaFree(sceneBox_);
  for (int k = 0; k < 2; ++k) {
    cudaFree(keys_[k]);
    cudaFree(vals_[k]);
  }
  cudaFree(nodeStorage_);
  cudaFree(flags_);
  cudaFree(reduceTemp_);
  cudaFree(sortTemp_);
  for (auto& e : events_) cudaEventDestroy(e);
  cudaStreamDestroy(stream_);
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

void Builder::setPoints(const Vec3* points) {
  // The stream is idle here: build() always waits for the GPU before returning.
  // For unified memory this write faults the pages back to the host.
  std::memcpy(hostPoints_, points, static_cast<size_t>(n_) * sizeof(Vec3));
}

// Host-to-device step. Kept out of the CUDA Graph so that all four memory
// modes work the same way in both launch modes.
void Builder::issueTransfer() {
  const size_t pointBytes = static_cast<size_t>(n_) * sizeof(Vec3);
  switch (hostMem_) {
    case HostMem::Pageable:
    case HostMem::Pinned:
      LBVH_CUDA_CHECK(cudaMemcpyAsync(devicePoints_, hostPoints_, pointBytes,
                                      cudaMemcpyHostToDevice, stream_));
      break;
    case HostMem::Unified:
      break;  // nothing to issue: the first kernel that touches a page migrates it
    case HostMem::UnifiedPrefetch: {
#if CUDART_VERSION >= 13000
      cudaMemLocation location;
      location.type = cudaMemLocationTypeDevice;
      location.id = device_;
      LBVH_CUDA_CHECK(cudaMemPrefetchAsync(devicePoints_, pointBytes, location, 0, stream_));
#else
      LBVH_CUDA_CHECK(cudaMemPrefetchAsync(devicePoints_, pointBytes, device_, stream_));
#endif
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// The five stages
// ---------------------------------------------------------------------------
// Everything goes into one stream, so the stages run in order. The same
// function is used for direct launches and for recording the CUDA Graph.
// With recordEvents, a CUDA event after each stage gives per-stage times.
void Builder::issueKernels(bool recordEvents) {
  auto mark = [&](int k) {
    if (recordEvents) LBVH_CUDA_CHECK(cudaEventRecord(events_[k], stream_));
  };
  const int block = LBVH_BLOCK_SIZE;

  // Stage 1: scene bounding box. The result stays on the GPU (sceneBox_), so
  // the host never has to wait for it.
  auto boxes = thrust::make_transform_iterator(devicePoints_, PointToBox());
  size_t reduceBytes = reduceTempBytes_;
  LBVH_CUDA_CHECK(cub::DeviceReduce::Reduce(reduceTemp_, reduceBytes, boxes, sceneBox_, n_,
                                            MergeBoxes(), emptyBox(), stream_));
  mark(2);

  // Stage 2: Morton codes.
  mortonKernel<<<gridSize(n_, block), block, 0, stream_>>>(devicePoints_, sceneBox_, keys_[0],
                                                           vals_[0], n_);
  LBVH_CUDA_CHECK(cudaGetLastError());
  mark(3);

  // Stage 3: sort (code, index) pairs by code. CUB ping-pongs between the two
  // buffers; Current() says which one holds the result. That is decided on
  // the host from the number of passes, so it is known immediately.
  cub::DoubleBuffer<uint32_t> keys(keys_[0], keys_[1]);
  cub::DoubleBuffer<uint32_t> vals(vals_[0], vals_[1]);
  size_t sortBytes = sortTempBytes_;
  LBVH_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sortTemp_, sortBytes, keys, vals, n_, 0,
                                                  kMortonBits, stream_));
  sortedKeys_ = keys.Current();
  sortedPrims_ = vals.Current();
  mark(4);

  // Stage 4: hierarchy. One thread per internal node.
  hierarchyKernel<<<gridSize(n_ - 1, block), block, 0, stream_>>>(sortedKeys_, n_, nodes_, flags_);
  LBVH_CUDA_CHECK(cudaGetLastError());
  mark(5);

  // Stage 5: refit. One thread per leaf.
  refitKernel<<<gridSize(n_, block), block, 0, stream_>>>(devicePoints_, sortedPrims_, n_, nodes_,
                                                          flags_);
  LBVH_CUDA_CHECK(cudaGetLastError());
  mark(6);
}

// Record the five stages once. Buffer addresses and n are baked into the
// graph, which is why a Builder is tied to one n.
void Builder::captureGraph() {
  const auto start = Clock::now();
  LBVH_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  issueKernels(false);
  LBVH_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
#if CUDART_VERSION >= 12000
  LBVH_CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, 0));
#else
  LBVH_CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
#endif
  LBVH_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &graphNodeCount_));
  graphSetupMs_ = msSince(start);
}

BuildTimes Builder::build(bool recordStageTimes) {
  const bool events = recordStageTimes && launch_ == LaunchMode::Separate;
  BuildTimes times;

  const auto start = Clock::now();
  if (events) LBVH_CUDA_CHECK(cudaEventRecord(events_[0], stream_));
  issueTransfer();
  if (events) LBVH_CUDA_CHECK(cudaEventRecord(events_[1], stream_));
  if (launch_ == LaunchMode::Graph) {
    LBVH_CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_));
  } else {
    issueKernels(events);
  }
  LBVH_CUDA_CHECK(cudaStreamSynchronize(stream_));
  times.totalMs = msSince(start);

  if (events) {
    float ms[kNumEvents - 1];
    for (int k = 0; k + 1 < kNumEvents; ++k)
      LBVH_CUDA_CHECK(cudaEventElapsedTime(&ms[k], events_[k], events_[k + 1]));
    times.transferMs = ms[0];
    times.boundsMs = ms[1];
    times.mortonMs = ms[2];
    times.sortMs = ms[3];
    times.hierarchyMs = ms[4];
    times.refitMs = ms[5];
  }
  return times;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

void Builder::download(HostTree& out) const {
  out.resize(n_);
  const size_t wordBytes = static_cast<size_t>(n_) * sizeof(uint32_t);
  const size_t numNodes = 2 * static_cast<size_t>(n_) - 1;
  LBVH_CUDA_CHECK(cudaMemcpy(&out.scene, sceneBox_, sizeof(AABB), cudaMemcpyDeviceToHost));
  LBVH_CUDA_CHECK(cudaMemcpy(out.keys.data(), sortedKeys_, wordBytes, cudaMemcpyDeviceToHost));
  LBVH_CUDA_CHECK(cudaMemcpy(out.prims.data(), sortedPrims_, wordBytes, cudaMemcpyDeviceToHost));
  std::vector<char> raw(nodeStorageBytes(numNodes));
  LBVH_CUDA_CHECK(cudaMemcpy(raw.data(), nodeStorage_, raw.size(), cudaMemcpyDeviceToHost));
  exportNodes(makeNodeView(raw.data(), numNodes), n_, out);
}

double Builder::radiusQuery(const std::vector<Vec3>& queries, float radius,
                            std::vector<int>& counts) {
  const int m = static_cast<int>(queries.size());
  counts.assign(m, 0);
  if (m == 0) return 0.0;
  Vec3* dQueries = nullptr;
  int* dCounts = nullptr;
  deviceAlloc(dQueries, m * sizeof(Vec3));
  deviceAlloc(dCounts, m * sizeof(int));
  LBVH_CUDA_CHECK(cudaMemcpy(dQueries, queries.data(), m * sizeof(Vec3), cudaMemcpyHostToDevice));
  LBVH_CUDA_CHECK(cudaStreamSynchronize(stream_));

  const auto start = Clock::now();
  radiusCountKernel<<<gridSize(m, kQueryBlockSize), kQueryBlockSize, 0, stream_>>>(
      nodes_, dQueries, m, radius * radius, dCounts);
  LBVH_CUDA_CHECK(cudaGetLastError());
  LBVH_CUDA_CHECK(cudaStreamSynchronize(stream_));
  const double ms = msSince(start);

  LBVH_CUDA_CHECK(cudaMemcpy(counts.data(), dCounts, m * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(dQueries);
  cudaFree(dCounts);
  return ms;
}

}  // namespace lbvh
