// kernels_impl.h -- the work one GPU thread does in each kernel.
//
// Each function below is the body of one kernel for one thread index i.
// kernels.cu wraps them in __global__ functions. tests/test_cpu.cpp calls the
// same functions on the CPU to check the logic without a GPU.
//
// Algorithm: T. Karras, "Maximizing Parallelism in the Construction of BVHs,
// Octrees, and k-d Trees", HPG 2012 (and "Thinking Parallel, Part III").
#pragma once

#include "lbvh/common.h"
#include "lbvh/nodes.h"

namespace lbvh {

// ---------------------------------------------------------------------------
// Portability helpers
// ---------------------------------------------------------------------------

// Number of leading zero bits; 32 for x == 0.
LBVH_HD LBVH_INLINE int clz32(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __clz(static_cast<int>(x));
#else
  return x ? __builtin_clz(x) : 32;
#endif
}

// Atomic post-increment: adds 1 and returns the old value.
LBVH_HD LBVH_INLINE int atomicFetchInc(int* p) {
#if defined(__CUDA_ARCH__)
  return atomicAdd(p, 1);
#else
  return __atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST);
#endif
}

// Make this thread's earlier stores visible to all other threads.
LBVH_HD LBVH_INLINE void publishStores() {
#if defined(__CUDA_ARCH__)
  __threadfence();
#endif
}

LBVH_HD LBVH_INLINE int leafIndex(int n, int k) { return n - 1 + k; }

// ---------------------------------------------------------------------------
// Stage 2: Morton codes
// ---------------------------------------------------------------------------
// Thread i reads point i, quantizes it inside the scene box and writes its
// Morton code and its own index (the value that the sort will carry along).
LBVH_HD LBVH_INLINE void mortonBody(int i, const Vec3* points, const AABB* sceneBox,
                                    uint32_t* keys, uint32_t* vals) {
  const Vec3 p = loadStream(points + i);
  const AABB scene = *sceneBox;  // same 24 bytes for every thread: stays in L1
  keys[i] = mortonCode(p, scene);
  vals[i] = static_cast<uint32_t>(i);
}

// ---------------------------------------------------------------------------
// Stage 4: hierarchy generation
// ---------------------------------------------------------------------------

// delta(i, j): length of the common prefix of sorted keys i and j, or -1 if j
// is out of range. When two keys are equal, the comparison falls back to the
// positions i and j themselves (Karras, Section 4), which behaves as if every
// key had its array index appended. All keys are then distinct, so duplicate
// Morton codes need no special handling anywhere else.
LBVH_HD LBVH_INLINE int delta(const uint32_t* keys, int n, int i, uint32_t ki, int j) {
  if (j < 0 || j >= n) return -1;
  const uint32_t kj = keys[j];
  if (ki == kj) return 32 + clz32(static_cast<uint32_t>(i ^ j));
  return clz32(ki ^ kj);
}

// Thread i builds internal node i (0 <= i <= n-2):
//   1. find the range of keys the node covers (node i sits at one end of it),
//   2. find where that range splits into two children,
//   3. write the child links, the children's parent links, and reset the
//      node's visit counter for the refit stage.
LBVH_HD LBVH_INLINE void hierarchyBody(int i, const uint32_t* keys, int n, const NodeView& nodes,
                                       int* flags) {
  const uint32_t ki = keys[i];

  // Direction of the range: towards the neighbour that shares more bits.
  const int dRight = delta(keys, n, i, ki, i + 1);
  const int dLeft = delta(keys, n, i, ki, i - 1);
  const int d = (dRight > dLeft) ? 1 : -1;

  // Every key inside the range shares more than deltaMin bits with key i.
  const int deltaMin = delta(keys, n, i, ki, i - d);

  // Exponential search for an upper bound on the range length...
  int lMax = 2;
  while (delta(keys, n, i, ki, i + lMax * d) > deltaMin) lMax <<= 1;

  // ...then binary search for the exact other end j.
  int l = 0;
  for (int t = lMax >> 1; t >= 1; t >>= 1) {
    if (delta(keys, n, i, ki, i + (l + t) * d) > deltaMin) l += t;
  }
  const int j = i + l * d;

  // Binary search for the split: the last key that shares more than
  // deltaNode bits with key i.
  const int deltaNode = delta(keys, n, i, ki, j);
  int s = 0;
  int t = l;
  do {
    t = (t + 1) >> 1;
    if (delta(keys, n, i, ki, i + (s + t) * d) > deltaNode) s += t;
  } while (t > 1);
  const int gamma = i + s * d + (d < 0 ? d : 0);

  // Children cover [first, gamma] and [gamma+1, last]. A child that covers a
  // single key is a leaf; otherwise it is the internal node with that index.
  const int first = (i < j) ? i : j;
  const int last = (i < j) ? j : i;
  const int leftChild = (first == gamma) ? leafIndex(n, gamma) : gamma;
  const int rightChild = (last == gamma + 1) ? leafIndex(n, gamma + 1) : gamma + 1;

  node::setChildren(nodes, i, leftChild, rightChild);
  node::setPrim(nodes, i, -1);
  node::setParent(nodes, leftChild, i);
  node::setParent(nodes, rightChild, i);
  if (i == 0) node::setParent(nodes, 0, -1);  // the root has no parent
  flags[i] = 0;
}

// ---------------------------------------------------------------------------
// Stage 5: bottom-up bounding-box refit
// ---------------------------------------------------------------------------
// Thread i starts at leaf i, writes the leaf, and walks towards the root.
// At each internal node an atomic counter decides who continues: the first
// thread to arrive stops, the second one merges both children's boxes and
// moves up. Every internal node is therefore processed exactly once, after
// both of its children are complete.
LBVH_HD LBVH_INLINE void refitBody(int i, int n, const Vec3* points, const uint32_t* sortedPrims,
                                   const NodeView& nodes, int* flags) {
  const int primIndex = static_cast<int>(loadStream(sortedPrims + i));
  const Vec3 p = loadStream(points + primIndex);

  int cur = leafIndex(n, i);
  AABB box = pointBox(p);
  node::setChildren(nodes, cur, -1, -1);
  node::setPrim(nodes, cur, primIndex);
  node::setBox(nodes, cur, box);

  int parent = node::parent(nodes, cur);
  while (parent >= 0) {
    // The box of `cur` must be visible to other threads before we announce
    // our arrival, because the sibling's thread may read it right after.
    publishStores();
    if (atomicFetchInc(&flags[parent]) == 0) return;  // first to arrive: stop

    // Second to arrive: the sibling subtree is complete and its box has been
    // published. Read it from L2 (not L1), merge, and store the parent's box.
    const int l = node::left(nodes, parent);
    const int r = node::right(nodes, parent);
    const int sibling = (l == cur) ? r : l;
    box = merge(box, node::boxShared(nodes, sibling));
    node::setBox(nodes, parent, box);

    cur = parent;
    parent = node::parent(nodes, cur);
  }
}

// ---------------------------------------------------------------------------
// Fixed query kernel (measurement only, not tuned)
// ---------------------------------------------------------------------------
// Counts the points within `radius` of q by walking down from the root with a
// small stack. Used to measure the build's share of application time and to
// check that a node layout that helps the build does not hurt queries.

LBVH_HD LBVH_INLINE float distanceSquared(const Vec3& q, const AABB& b) {
  const float dx = fmaxf(fmaxf(b.lo.x - q.x, q.x - b.hi.x), 0.0f);
  const float dy = fmaxf(fmaxf(b.lo.y - q.y, q.y - b.hi.y), 0.0f);
  const float dz = fmaxf(fmaxf(b.lo.z - q.z, q.z - b.hi.z), 0.0f);
  return dx * dx + dy * dy + dz * dz;
}

// Deepest possible tree: 30 key bits plus up to 31 tie-break bits.
constexpr int kQueryStackSize = 64;

LBVH_HD LBVH_INLINE int radiusCountBody(const NodeView& nodes, const Vec3& q, float radiusSquared) {
  int stack[kQueryStackSize];
  int top = 0;
  int count = 0;
  stack[top++] = 0;  // root
  while (top > 0) {
    const int k = stack[--top];
    if (distanceSquared(q, node::box(nodes, k)) > radiusSquared) continue;
    const int l = node::left(nodes, k);
    if (l < 0) {  // leaf: its box is the point itself
      ++count;
      continue;
    }
    stack[top++] = l;
    stack[top++] = node::right(nodes, k);
  }
  return count;
}

}  // namespace lbvh
