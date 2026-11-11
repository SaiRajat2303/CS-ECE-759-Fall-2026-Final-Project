// tree.h -- a layout-independent copy of a tree on the host, the independent
// CPU reference builder, and the comparison used to verify every GPU build.
// Plain C++: no CUDA needed.
#pragma once

#include <string>
#include <vector>

#include "lbvh/common.h"

namespace lbvh {

struct HostTree {
  int n = 0;                     // number of primitives
  AABB scene{};                  // bounding box of all points
  std::vector<uint32_t> keys;    // n sorted Morton codes
  std::vector<uint32_t> prims;   // n input indices in Morton order
  // Per node (2n-1 entries; internal nodes first, then leaves):
  std::vector<int> parent, left, right, prim;
  std::vector<AABB> box;

  void resize(int numPrims) {
    n = numPrims;
    keys.resize(n);
    prims.resize(n);
    const size_t numNodes = 2 * static_cast<size_t>(n) - 1;
    parent.resize(numNodes);
    left.resize(numNodes);
    right.resize(numNodes);
    prim.resize(numNodes);
    box.resize(numNodes);
  }
};

// Builds the reference tree on the CPU with a top-down algorithm that shares
// no search code with the GPU kernels (only the Morton-code formula).
void buildReference(const Vec3* points, int n, HostTree& out);

// True if both trees are identical: scene box, sorted keys, primitive order,
// every link and every bounding box. On mismatch, `why` names the first
// difference found.
bool compareTrees(const HostTree& got, const HostTree& want, std::string& why);

// Structural check that does not need a reference: every leaf is reached
// exactly once from the root, links are consistent, boxes enclose children.
bool validateTree(const HostTree& t, std::string& why);

// Cheap descriptors of an input, computed from its sorted Morton codes.
// These are the candidate input features for the learned cost model.
struct InputFeatures {
  double extentX = 0, extentY = 0, extentZ = 0;
  double duplicateFraction = 0;   // keys equal to their predecessor / n
  double occupiedCellFraction = 0;  // occupied cells of a 32^3 grid / 32768
  double maxCellFraction = 0;     // points in the fullest 32^3 cell / n
};
InputFeatures computeFeatures(const HostTree& t);

}  // namespace lbvh
