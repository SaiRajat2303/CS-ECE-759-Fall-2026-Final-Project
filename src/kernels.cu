// kernels.cu -- the GPU kernels. Each one maps a thread to an index and calls
// the matching body from kernels_impl.h.
//
// Compiled separately from the rest so that -maxrregcount (the register-cap
// knob) only affects these kernels. Compile-time knobs used here:
//   LBVH_BLOCK_SIZE       threads per block, also given to __launch_bounds__
//   LBVH_LAYOUT_SOA       node layout (through NodeView and its accessors)
//   LBVH_STREAM_HINT_CG   cache policy of read-once loads (through loadStream)
#include "lbvh/kernels.h"
#include "lbvh/kernels_impl.h"

namespace lbvh {

extern const int LBVH_ABI_TAG = 1;

__global__ void __launch_bounds__(LBVH_BLOCK_SIZE)
mortonKernel(const Vec3* points, const AABB* sceneBox, uint32_t* keys, uint32_t* vals, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mortonBody(i, points, sceneBox, keys, vals);
}

__global__ void __launch_bounds__(LBVH_BLOCK_SIZE)
hierarchyKernel(const uint32_t* sortedKeys, int n, NodeView nodes, int* flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n - 1) hierarchyBody(i, sortedKeys, n, nodes, flags);
}

__global__ void __launch_bounds__(LBVH_BLOCK_SIZE)
refitKernel(const Vec3* points, const uint32_t* sortedPrims, int n, NodeView nodes, int* flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) refitBody(i, n, points, sortedPrims, nodes, flags);
}

__global__ void __launch_bounds__(kQueryBlockSize)
radiusCountKernel(NodeView nodes, const Vec3* queries, int numQueries, float radiusSquared,
                  int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numQueries) counts[i] = radiusCountBody(nodes, queries[i], radiusSquared);
}

}  // namespace lbvh
