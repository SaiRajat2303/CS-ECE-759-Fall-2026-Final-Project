// export.h -- copy nodes from either layout into a layout-independent HostTree.
#pragma once

#include "lbvh/nodes.h"
#include "lbvh/tree.h"

namespace lbvh {

// `view` must point at host memory holding 2n-1 nodes. `out` must already be
// resized to n; keys, prims and scene are filled by the caller.
inline void exportNodes(const NodeView& view, int n, HostTree& out) {
  const int numNodes = 2 * n - 1;
  for (int k = 0; k < numNodes; ++k) {
    out.parent[k] = node::parent(view, k);
    out.left[k] = node::left(view, k);
    out.right[k] = node::right(view, k);
    out.prim[k] = node::prim(view, k);
    out.box[k] = node::box(view, k);
  }
}

}  // namespace lbvh
