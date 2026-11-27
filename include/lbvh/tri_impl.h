// tri_impl.h -- per-thread logic for an LBVH over triangles, and closest-hit
// ray traversal of the finished tree.
//
// The point-cloud pipeline in kernels_impl.h is reused unchanged for the
// stages that only need one point per primitive (Morton codes of triangle
// centroids, sort, hierarchy). What differs for triangles is:
//   - a primitive has a box, not a point: triBoundsBody computes it,
//   - leaves get that box in the refit stage: refitTriBody,
//   - rays are traced against the tree: traceClosestBody.
//
// Like kernels_impl.h, everything here compiles as CUDA device code and as
// plain C++, so the same functions can be unit-tested on the CPU.
#pragma once

#include "lbvh/common.h"
#include "lbvh/kernels_impl.h"
#include "lbvh/nodes.h"

namespace lbvh {

struct Tri {
  int a, b, c;  // vertex indices
};

// ---------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------
LBVH_HD LBVH_INLINE Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
LBVH_HD LBVH_INLINE float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
LBVH_HD LBVH_INLINE Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

LBVH_HD LBVH_INLINE float surfaceArea(const AABB& b) {
  const float dx = b.hi.x - b.lo.x, dy = b.hi.y - b.lo.y, dz = b.hi.z - b.lo.z;
  if (dx < 0.0f || dy < 0.0f || dz < 0.0f) return 0.0f;  // empty box
  return 2.0f * (dx * dy + dy * dz + dz * dx);
}

// ---------------------------------------------------------------------------
// Stage 0: triangle boxes and centroids
// ---------------------------------------------------------------------------
// Thread i writes the box of triangle i and the centre of that box. The
// centres are what the Morton stage sorts; the boxes become the leaf boxes.
LBVH_HD LBVH_INLINE void triBoundsBody(int i, const Vec3* verts, const Tri* tris, AABB* boxes,
                                       Vec3* centroids) {
  const Tri t = tris[i];
  const Vec3 a = loadStream(verts + t.a);
  const Vec3 b = loadStream(verts + t.b);
  const Vec3 c = loadStream(verts + t.c);
  const AABB box = merge(merge(pointBox(a), pointBox(b)), pointBox(c));
  boxes[i] = box;
  centroids[i] = {0.5f * (box.lo.x + box.hi.x), 0.5f * (box.lo.y + box.hi.y),
                  0.5f * (box.lo.z + box.hi.z)};
}

// ---------------------------------------------------------------------------
// Stage 5 (triangles): bottom-up refit with triangle boxes at the leaves
// ---------------------------------------------------------------------------
// Same hand-off protocol as refitBody: the second thread to reach an internal
// node merges both children and continues. flags[0..n-2] must be zero on
// entry (hierarchyBody resets them; an update without a rebuild must clear
// them itself).
//
// The topology (parent/left/right/prim) is not changed, so this body is also
// the whole "update" step for an animated mesh: recompute the triangle boxes
// for the new vertex positions, clear the flags, refit.
LBVH_HD LBVH_INLINE void refitTriBody(int i, int n, const AABB* triBoxes,
                                      const uint32_t* sortedPrims, const NodeView& nodes,
                                      int* flags) {
  const int primIndex = static_cast<int>(loadStream(sortedPrims + i));
  AABB box = triBoxes[primIndex];

  int cur = leafIndex(n, i);
  node::setChildren(nodes, cur, -1, -1);
  node::setPrim(nodes, cur, primIndex);
  node::setBox(nodes, cur, box);

  int parent = node::parent(nodes, cur);
  while (parent >= 0) {
    publishStores();
    if (atomicFetchInc(&flags[parent]) == 0) return;

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
// Ray queries
// ---------------------------------------------------------------------------

struct RayQuery {
  Vec3 o, d;
  float tmin, tmax;
};

struct TriHit {
  float t;    // INFINITY on miss
  int prim;   // -1 on miss
  float u, v; // barycentrics of vertices b and c
};

// Slab test. Returns the entry distance, or INFINITY if the ray misses the box
// inside [tmin, tmax]. The far distance is scaled up by 1 + 2*gamma(3)
// (Ize, "Robust BVH Ray Traversal", JCGT 2013) so that rounding cannot make a
// ray that grazes a box face miss a triangle lying in that face.
LBVH_HD LBVH_INLINE float rayBoxEntry(const Vec3& o, const Vec3& invD, const AABB& b, float tmin,
                                      float tmax) {
  const float tx0 = (b.lo.x - o.x) * invD.x, tx1 = (b.hi.x - o.x) * invD.x;
  const float ty0 = (b.lo.y - o.y) * invD.y, ty1 = (b.hi.y - o.y) * invD.y;
  const float tz0 = (b.lo.z - o.z) * invD.z, tz1 = (b.hi.z - o.z) * invD.z;
  const float tNear = fmaxf(fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)), fmaxf(fminf(tz0, tz1), tmin));
  float tFar = fminf(fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)), fminf(fmaxf(tz0, tz1), tmax));
  tFar *= 1.00000036f;
  return tNear <= tFar ? tNear : INFINITY;
}

// Moller-Trumbore. Updates `hit` if triangle `prim` is hit closer than hit.t.
LBVH_HD LBVH_INLINE void intersectTri(const RayQuery& r, const Vec3* verts, const Tri* tris,
                                      int prim, TriHit& hit) {
  const Tri t = tris[prim];
  const Vec3 a = verts[t.a], b = verts[t.b], c = verts[t.c];
  const Vec3 e1 = sub(b, a), e2 = sub(c, a);
  const Vec3 p = cross(r.d, e2);
  const float det = dot(e1, p);
  if (det == 0.0f) return;  // ray parallel to the triangle (or degenerate triangle)
  const float inv = 1.0f / det;
  const Vec3 s = sub(r.o, a);
  const float u = dot(s, p) * inv;
  if (u < 0.0f || u > 1.0f) return;
  const Vec3 q = cross(s, e1);
  const float v = dot(r.d, q) * inv;
  if (v < 0.0f || u + v > 1.0f) return;
  const float tt = dot(e2, q) * inv;
  if (tt < r.tmin || tt > r.tmax || tt >= hit.t) return;
  hit = TriHit{tt, prim, u, v};
}

// Deepest possible tree: 30 key bits plus up to 31 tie-break bits.
constexpr int kTraceStackSize = 64;

// Closest hit of one ray. Front-to-back: at each internal node both child
// boxes are tested, the far child is pushed and the near child is visited
// next. Entries carry their box-entry distance so that subtrees behind the
// current closest hit are skipped when popped.
//
// If `visits` is non-null it receives the number of nodes whose box was
// tested (the "nodes visited" metric of the benchmark).
LBVH_HD LBVH_INLINE TriHit traceClosestBody(const NodeView& nodes, int n, const Vec3* verts,
                                            const Tri* tris, RayQuery r, int* visits = nullptr) {
  TriHit hit{INFINITY, -1, 0.0f, 0.0f};
  const Vec3 invD = {1.0f / r.d.x, 1.0f / r.d.y, 1.0f / r.d.z};
  int count = 1;

  // n == 1: the root is the only leaf.
  if (n == 1) {
    if (rayBoxEntry(r.o, invD, node::box(nodes, 0), r.tmin, r.tmax) < INFINITY)
      intersectTri(r, verts, tris, node::prim(nodes, 0), hit);
    if (visits) *visits = count;
    return hit;
  }

  if (rayBoxEntry(r.o, invD, node::box(nodes, 0), r.tmin, r.tmax) == INFINITY) {
    if (visits) *visits = count;
    return hit;
  }

  int stackNode[kTraceStackSize];
  float stackT[kTraceStackSize];
  int top = 0;
  int k = 0;  // current internal node; its box is known to be hit

  while (true) {
    const int l = node::left(nodes, k);
    const int rr = node::right(nodes, k);
    const float far = hit.t < r.tmax ? hit.t : r.tmax;
    float tl = rayBoxEntry(r.o, invD, node::box(nodes, l), r.tmin, far);
    float tr = rayBoxEntry(r.o, invD, node::box(nodes, rr), r.tmin, far);
    count += 2;

    // Leaves are intersected right away instead of being pushed.
    if (tl < INFINITY && node::left(nodes, l) < 0) {
      intersectTri(r, verts, tris, node::prim(nodes, l), hit);
      tl = INFINITY;
      if (tr > hit.t) tr = INFINITY;
    }
    if (tr < INFINITY && node::left(nodes, rr) < 0) {
      intersectTri(r, verts, tris, node::prim(nodes, rr), hit);
      tr = INFINITY;
      if (tl > hit.t) tl = INFINITY;
    }

    if (tl < INFINITY && tr < INFINITY) {
      // Visit the nearer child next, keep the other one for later.
      const bool leftFirst = tl <= tr;
      stackNode[top] = leftFirst ? rr : l;
      stackT[top] = leftFirst ? tr : tl;
      ++top;
      k = leftFirst ? l : rr;
      continue;
    }
    if (tl < INFINITY) { k = l; continue; }
    if (tr < INFINITY) { k = rr; continue; }

    // Both children done: pop the nearest pending subtree still in front of
    // the closest hit.
    bool found = false;
    while (top > 0) {
      --top;
      if (stackT[top] <= hit.t) {
        k = stackNode[top];
        found = true;
        break;
      }
    }
    if (!found) break;
  }

  if (visits) *visits = count;
  return hit;
}

// ---------------------------------------------------------------------------
// SAH cost of a finished tree (diagnostic only)
// ---------------------------------------------------------------------------
// Per-node contribution, before normalisation by the root's surface area:
// kSahTraversal * SA(node) for internal nodes, kSahIntersect * SA(leaf) for
// leaves (one triangle per leaf).
constexpr float kSahTraversal = 1.2f;
constexpr float kSahIntersect = 1.0f;

LBVH_HD LBVH_INLINE double sahNodeCost(const NodeView& nodes, int k) {
  const double sa = surfaceArea(node::box(nodes, k));
  return (node::left(nodes, k) < 0 ? kSahIntersect : kSahTraversal) * sa;
}

}  // namespace lbvh
