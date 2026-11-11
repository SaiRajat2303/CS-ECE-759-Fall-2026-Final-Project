// reference.cpp -- independent CPU reference builder and tree checks.
//
// The reference builds the same binary radix tree as the GPU kernels, but
// top-down: each range of sorted keys is split at its highest differing bit.
// It shares only the Morton-code formula with the GPU code, so agreement
// between the two is real evidence that the GPU build is correct.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <sstream>

#include "lbvh/tree.h"

namespace lbvh {

namespace {

// Key i extended with its position, so all keys are distinct. This is the
// explicit form of the tie-break that the GPU delta() function applies.
inline uint64_t augmentedKey(const std::vector<uint32_t>& keys, int i) {
  return (static_cast<uint64_t>(keys[i]) << 32) | static_cast<uint32_t>(i);
}

}  // namespace

void buildReference(const Vec3* points, int n, HostTree& out) {
  out.resize(n);

  // Stage 1: scene bounding box.
  AABB scene = emptyBox();
  for (int i = 0; i < n; ++i) scene = merge(scene, pointBox(points[i]));
  out.scene = scene;

  // Stage 2: Morton codes.
  std::vector<uint32_t> codes(n);
  for (int i = 0; i < n; ++i) codes[i] = mortonCode(points[i], scene);

  // Stage 3: stable sort by code (radix sort on the GPU is stable too, so
  // points with equal codes stay in input order in both).
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(),
                   [&](uint32_t a, uint32_t b) { return codes[a] < codes[b]; });
  for (int i = 0; i < n; ++i) {
    out.keys[i] = codes[order[i]];
    out.prims[i] = order[i];
  }

  // Leaves.
  const int leafBase = n - 1;
  for (int k = 0; k < n; ++k) {
    const int id = leafBase + k;
    out.left[id] = -1;
    out.right[id] = -1;
    out.prim[id] = static_cast<int>(order[k]);
    out.box[id] = pointBox(points[order[k]]);
  }
  if (n == 1) {
    out.parent[0] = -1;
    return;
  }

  // Stage 4: hierarchy, top-down with an explicit stack.
  struct Range {
    int nodeId, first, last;
  };
  std::vector<Range> stack;
  std::vector<int> visitOrder;  // internal nodes, parents before children
  visitOrder.reserve(n - 1);
  out.parent[0] = -1;
  stack.push_back({0, 0, n - 1});
  while (!stack.empty()) {
    const Range r = stack.back();
    stack.pop_back();
    visitOrder.push_back(r.nodeId);

    // Highest bit in which the first and last key of the range differ.
    const uint64_t diff = augmentedKey(out.keys, r.first) ^ augmentedKey(out.keys, r.last);
    const int bit = 63 - __builtin_clzll(diff);
    // Keys are sorted, so that bit is 0 for a prefix of the range and 1 for
    // the rest. Find the last key where it is 0.
    int lo = r.first, hi = r.last;
    while (hi - lo > 1) {
      const int mid = lo + (hi - lo) / 2;
      if ((augmentedKey(out.keys, mid) >> bit) & 1ull) hi = mid; else lo = mid;
    }
    const int split = lo;

    const int leftId = (split == r.first) ? leafBase + split : split;
    const int rightId = (split + 1 == r.last) ? leafBase + split + 1 : split + 1;
    out.left[r.nodeId] = leftId;
    out.right[r.nodeId] = rightId;
    out.prim[r.nodeId] = -1;
    out.parent[leftId] = r.nodeId;
    out.parent[rightId] = r.nodeId;
    if (split != r.first) stack.push_back({split, r.first, split});
    if (split + 1 != r.last) stack.push_back({split + 1, split + 1, r.last});
  }

  // Stage 5: boxes, children before parents.
  for (auto it = visitOrder.rbegin(); it != visitOrder.rend(); ++it) {
    const int id = *it;
    out.box[id] = merge(out.box[out.left[id]], out.box[out.right[id]]);
  }
}

namespace {

inline bool sameBox(const AABB& a, const AABB& b) {
  return a.lo.x == b.lo.x && a.lo.y == b.lo.y && a.lo.z == b.lo.z &&
         a.hi.x == b.hi.x && a.hi.y == b.hi.y && a.hi.z == b.hi.z;
}

template <typename T>
bool firstMismatch(const std::vector<T>& got, const std::vector<T>& want, const char* name,
                   std::string& why) {
  if (got.size() != want.size()) {
    why = std::string(name) + ": size differs";
    return true;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      std::ostringstream os;
      os << name << "[" << i << "]: got " << got[i] << ", want " << want[i];
      why = os.str();
      return true;
    }
  }
  return false;
}

}  // namespace

bool compareTrees(const HostTree& got, const HostTree& want, std::string& why) {
  if (got.n != want.n) {
    why = "n differs";
    return false;
  }
  if (!sameBox(got.scene, want.scene)) {
    why = "scene box differs";
    return false;
  }
  if (firstMismatch(got.keys, want.keys, "keys", why)) return false;
  if (firstMismatch(got.prims, want.prims, "prims", why)) return false;
  if (firstMismatch(got.parent, want.parent, "parent", why)) return false;
  if (firstMismatch(got.left, want.left, "left", why)) return false;
  if (firstMismatch(got.right, want.right, "right", why)) return false;
  if (firstMismatch(got.prim, want.prim, "prim", why)) return false;
  for (size_t i = 0; i < want.box.size(); ++i) {
    if (!sameBox(got.box[i], want.box[i])) {
      std::ostringstream os;
      os << "box[" << i << "] differs";
      why = os.str();
      return false;
    }
  }
  return true;
}

bool validateTree(const HostTree& t, std::string& why) {
  const int n = t.n;
  const int numNodes = 2 * n - 1;
  const int leafBase = n - 1;
  const int root = 0;  // also correct for n == 1, where the only node is a leaf
  if (t.parent[root] != -1) {
    why = "root has a parent";
    return false;
  }
  std::vector<char> seen(numNodes, 0);
  std::vector<char> primSeen(n, 0);
  std::vector<int> stack{root};
  int leaves = 0;
  while (!stack.empty()) {
    const int id = stack.back();
    stack.pop_back();
    if (id < 0 || id >= numNodes || seen[id]) {
      why = "node visited twice or out of range";
      return false;
    }
    seen[id] = 1;
    if (id >= leafBase) {  // leaf
      const int p = t.prim[id];
      if (t.left[id] != -1 || t.right[id] != -1 || p < 0 || p >= n || primSeen[p]) {
        why = "bad leaf";
        return false;
      }
      primSeen[p] = 1;
      ++leaves;
      continue;
    }
    const int l = t.left[id], r = t.right[id];
    if (l < 0 || l >= numNodes || r < 0 || r >= numNodes || t.parent[l] != id || t.parent[r] != id) {
      why = "bad child or parent link";
      return false;
    }
    const AABB m = merge(t.box[l], t.box[r]);
    if (!sameBox(m, t.box[id])) {
      why = "box is not the union of its children";
      return false;
    }
    stack.push_back(l);
    stack.push_back(r);
  }
  if (leaves != n) {
    why = "not every leaf is reachable";
    return false;
  }
  return true;
}

InputFeatures computeFeatures(const HostTree& t) {
  InputFeatures f;
  f.extentX = static_cast<double>(t.scene.hi.x) - t.scene.lo.x;
  f.extentY = static_cast<double>(t.scene.hi.y) - t.scene.lo.y;
  f.extentZ = static_cast<double>(t.scene.hi.z) - t.scene.lo.z;
  const int n = t.n;
  if (n == 0) return f;
  // The top 15 bits of a 30-bit code identify a cell of a 32x32x32 grid, and
  // sorted codes keep each cell's points together.
  const int shift = kMortonBits - 15;
  long duplicates = 0, cells = 0, run = 0, maxRun = 0;
  for (int i = 0; i < n; ++i) {
    if (i > 0 && t.keys[i] == t.keys[i - 1]) ++duplicates;
    if (i == 0 || (t.keys[i] >> shift) != (t.keys[i - 1] >> shift)) {
      ++cells;
      run = 0;
    }
    ++run;
    if (run > maxRun) maxRun = run;
  }
  f.duplicateFraction = static_cast<double>(duplicates) / n;
  f.occupiedCellFraction = static_cast<double>(cells) / 32768.0;
  f.maxCellFraction = static_cast<double>(maxRun) / n;
  return f;
}

}  // namespace lbvh
