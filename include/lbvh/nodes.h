// nodes.h -- BVH node storage in two layouts behind one accessor interface.
//
// A tree over n primitives has 2n-1 nodes:
//   internal nodes  0 .. n-2      (node 0 is the root)
//   leaf nodes      n-1 .. 2n-2   (leaf k is node n-1+k and holds the k-th
//                                  primitive in Morton order)
//
// Each node has 10 four-byte fields (40 bytes):
//   parent, left, right, prim, and the box lox, loy, loz, hix, hiy, hiz.
//   Leaves have left = right = -1 and prim = index of the input point.
//   Internal nodes have prim = -1.
//
// AoS keeps the 10 fields of a node together. SoA keeps 10 separate arrays.
// Both use the same total memory, carved out of one allocation.
#pragma once

#include "lbvh/common.h"

namespace lbvh {

struct NodeAoS {
  int parent, left, right, prim;
  float lox, loy, loz, hix, hiy, hiz;
};
static_assert(sizeof(NodeAoS) == 40, "NodeAoS must be 40 bytes");

// A view of the node storage. Passed by value to kernels.
struct NodeView {
#if LBVH_LAYOUT_SOA
  int *parent, *left, *right, *prim;
  float *lox, *loy, *loz, *hix, *hiy, *hiz;
#else
  NodeAoS* n;
#endif
};

inline size_t nodeStorageBytes(size_t numNodes) { return numNodes * sizeof(NodeAoS); }

// Build a view over a block of nodeStorageBytes(numNodes) bytes.
inline NodeView makeNodeView(void* base, size_t numNodes) {
  NodeView v;
#if LBVH_LAYOUT_SOA
  int* ints = static_cast<int*>(base);
  v.parent = ints;
  v.left = ints + numNodes;
  v.right = ints + 2 * numNodes;
  v.prim = ints + 3 * numNodes;
  float* floats = reinterpret_cast<float*>(ints + 4 * numNodes);
  v.lox = floats;
  v.loy = floats + numNodes;
  v.loz = floats + 2 * numNodes;
  v.hix = floats + 3 * numNodes;
  v.hiy = floats + 4 * numNodes;
  v.hiz = floats + 5 * numNodes;
#else
  (void)numNodes;
  v.n = static_cast<NodeAoS*>(base);
#endif
  return v;
}

// One spelling for "field f of node i" in both layouts.
#if LBVH_LAYOUT_SOA
#define LBVH_FIELD(view, f, i) ((view).f[(i)])
#else
#define LBVH_FIELD(view, f, i) ((view).n[(i)].f)
#endif

namespace node {

LBVH_HD LBVH_INLINE int parent(const NodeView& v, int i) { return LBVH_FIELD(v, parent, i); }
LBVH_HD LBVH_INLINE int left(const NodeView& v, int i) { return LBVH_FIELD(v, left, i); }
LBVH_HD LBVH_INLINE int right(const NodeView& v, int i) { return LBVH_FIELD(v, right, i); }
LBVH_HD LBVH_INLINE int prim(const NodeView& v, int i) { return LBVH_FIELD(v, prim, i); }

LBVH_HD LBVH_INLINE void setParent(const NodeView& v, int i, int p) { LBVH_FIELD(v, parent, i) = p; }
LBVH_HD LBVH_INLINE void setPrim(const NodeView& v, int i, int p) { LBVH_FIELD(v, prim, i) = p; }
LBVH_HD LBVH_INLINE void setChildren(const NodeView& v, int i, int l, int r) {
  LBVH_FIELD(v, left, i) = l;
  LBVH_FIELD(v, right, i) = r;
}

LBVH_HD LBVH_INLINE void setBox(const NodeView& v, int i, const AABB& b) {
  LBVH_FIELD(v, lox, i) = b.lo.x;
  LBVH_FIELD(v, loy, i) = b.lo.y;
  LBVH_FIELD(v, loz, i) = b.lo.z;
  LBVH_FIELD(v, hix, i) = b.hi.x;
  LBVH_FIELD(v, hiy, i) = b.hi.y;
  LBVH_FIELD(v, hiz, i) = b.hi.z;
}

// Box load with the default cache policy. Safe once the build has finished.
LBVH_HD LBVH_INLINE AABB box(const NodeView& v, int i) {
  AABB b;
  b.lo.x = LBVH_FIELD(v, lox, i);
  b.lo.y = LBVH_FIELD(v, loy, i);
  b.lo.z = LBVH_FIELD(v, loz, i);
  b.hi.x = LBVH_FIELD(v, hix, i);
  b.hi.y = LBVH_FIELD(v, hiy, i);
  b.hi.z = LBVH_FIELD(v, hiz, i);
  return b;
}

// Box load that bypasses L1. Required during refit, where the box may have
// been written by a thread on another SM a moment ago.
LBVH_HD LBVH_INLINE AABB boxShared(const NodeView& v, int i) {
  AABB b;
  b.lo.x = loadShared(&LBVH_FIELD(v, lox, i));
  b.lo.y = loadShared(&LBVH_FIELD(v, loy, i));
  b.lo.z = loadShared(&LBVH_FIELD(v, loz, i));
  b.hi.x = loadShared(&LBVH_FIELD(v, hix, i));
  b.hi.y = loadShared(&LBVH_FIELD(v, hiy, i));
  b.hi.z = loadShared(&LBVH_FIELD(v, hiz, i));
  return b;
}

}  // namespace node
}  // namespace lbvh
