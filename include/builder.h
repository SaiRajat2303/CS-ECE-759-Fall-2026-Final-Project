// builder.h -- host-side driver of the LBVH build.
//
// A Builder owns every buffer the build needs for a fixed number of points n
// and runs the five-stage pipeline:
//   1. scene bounding box   (cub::DeviceReduce)
//   2. Morton codes         (mortonKernel)
//   3. radix sort           (cub::DeviceRadixSort)
//   4. hierarchy generation (hierarchyKernel)
//   5. bounding-box refit   (refitKernel)
//
// Runtime knobs: how the input points reach the GPU (HostMem) and how the
// kernels are launched (LaunchMode). Compile-time knobs: see common.h.
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "lbvh/common.h"
#include "lbvh/nodes.h"
#include "lbvh/tree.h"

namespace lbvh {

enum class HostMem {
  Pageable,         // malloc + cudaMemcpyAsync
  Pinned,           // cudaHostAlloc (page-locked) + cudaMemcpyAsync
  Unified,          // cudaMallocManaged, pages migrate on first GPU access
  UnifiedPrefetch,  // cudaMallocManaged + cudaMemPrefetchAsync before the build
};

enum class LaunchMode {
  Separate,  // one launch call per kernel
  Graph,     // kernels recorded once into a CUDA Graph, replayed with one call
};

const char* hostMemName(HostMem m);
const char* launchModeName(LaunchMode m);
bool parseHostMem(const std::string& s, HostMem& m);
bool parseLaunchMode(const std::string& s, LaunchMode& m);

struct BuildTimes {
  double totalMs = 0;  // wall clock: from the first call until the GPU is done
  // Per-stage GPU times from CUDA events. Only filled when build(true) is
  // used in Separate launch mode; otherwise -1.
  double transferMs = -1, boundsMs = -1, mortonMs = -1, sortMs = -1, hierarchyMs = -1,
         refitMs = -1;
};

class Builder {
 public:
  Builder(int n, HostMem hostMem, LaunchMode launch);
  ~Builder();
  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

  // Copies n points into the builder's host-side buffer. Call before every
  // build(): it stands for "the application produced a new frame of points on
  // the CPU" and is not part of the timed build. For unified memory it also
  // brings the pages back to the host, so each build pays the migration again.
  void setPoints(const Vec3* points);

  // Runs transfer + the five stages and waits for the GPU to finish.
  BuildTimes build(bool recordStageTimes = false);

  // Copies the finished tree to the host in a layout-independent form.
  void download(HostTree& out) const;

  // Runs the fixed query kernel: counts[i] = points within `radius` of
  // queries[i]. Returns the kernel time in milliseconds (copies excluded).
  double radiusQuery(const std::vector<Vec3>& queries, float radius, std::vector<int>& counts);

  int size() const { return n_; }
  NodeView nodes() const { return nodes_; }            // device memory
  const uint32_t* sortedKeys() const { return sortedKeys_; }    // device memory
  const uint32_t* sortedPrims() const { return sortedPrims_; }  // device memory
  size_t deviceBytes() const { return deviceBytes_; }

  // CUDA Graph details (0 in Separate mode).
  double graphSetupMs() const { return graphSetupMs_; }
  size_t graphNodeCount() const { return graphNodeCount_; }

 private:
  void issueTransfer();
  void issueKernels(bool recordEvents);
  void captureGraph();

  int n_;
  HostMem hostMem_;
  LaunchMode launch_;
  int device_ = 0;
  cudaStream_t stream_ = nullptr;

  Vec3* hostPoints_ = nullptr;    // pageable, pinned or managed
  Vec3* devicePoints_ = nullptr;  // device copy, or the managed buffer itself
  AABB* sceneBox_ = nullptr;
  uint32_t* keys_[2] = {nullptr, nullptr};
  uint32_t* vals_[2] = {nullptr, nullptr};
  uint32_t* sortedKeys_ = nullptr;
  uint32_t* sortedPrims_ = nullptr;
  void* nodeStorage_ = nullptr;
  NodeView nodes_{};
  int* flags_ = nullptr;
  void* reduceTemp_ = nullptr;
  size_t reduceTempBytes_ = 0;
  void* sortTemp_ = nullptr;
  size_t sortTempBytes_ = 0;
  size_t deviceBytes_ = 0;

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graphExec_ = nullptr;
  double graphSetupMs_ = 0;
  size_t graphNodeCount_ = 0;

  static constexpr int kNumEvents = 7;
  cudaEvent_t events_[kNumEvents] = {};
};

}  // namespace lbvh
