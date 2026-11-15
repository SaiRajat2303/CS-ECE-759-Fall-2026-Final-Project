#pragma once
#include "lbvh/common.h"
#include "ply.h"
#include <vector>

/* Same expression on host and device. An add then a multiply cannot be fused, so both sides round identically. */
LBVH_HD LBVH_INLINE lbvh::Vec3 box_centroid(const lbvh::AABB& b) {
    return {(b.lo.x + b.hi.x) * 0.5f, (b.lo.y + b.hi.y) * 0.5f, (b.lo.z + b.hi.z) * 0.5f};
}

inline std::vector<lbvh::AABB> triangle_boxes(const PlyMesh& m) {
    std::vector<lbvh::AABB> out(m.triangles());
    for (size_t t = 0; t < out.size(); ++t) {
        lbvh::AABB b = lbvh::emptyBox();
        for (int k = 0; k < 3; ++k) {
            const float* v = &m.xyz[3 * (size_t)m.tris[3 * t + k]];
            b = lbvh::merge(b, lbvh::pointBox({v[0], v[1], v[2]}));
        }
        out[t] = b;
    }
    return out;
}
