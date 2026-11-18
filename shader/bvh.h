#pragma once

#include <cstddef>
#include <vector>

#include "common.h"
#include "mesh.h"

namespace Shader
{

  struct BVHNode {
    AABB bounds;
    int left = -1, right = -1;
    int first = 0, count = 0;
    bool isLeaf() const { return count > 0; }
  };

  class BVH {
    public:
    explicit BVH(const Mesh::Mesh& mesh);

    // Closest-hit query
    bool intersect(const Ray& ray, float tMin, float tMax, Mesh::HitRecord& rec) const;
    // Any-hit query: stops at the first hit
    bool occluded(const Ray& ray, float tMin, float tMax) const;

    size_t nodeCount() const { return nodes_.size(); }

    private:
    static constexpr int kMaxLeafSize = 4;
    static constexpr int kSahBuckets = 12;

    const Mesh::Mesh& mesh_;
    std::vector<int> triIndices_;
    std::vector<AABB> triBounds_;
    std::vector<Vec3> centroids_;
    std::vector<BVHNode> nodes_;

    bool traverse(const Ray& ray, float tMin, float tMax, Mesh::HitRecord* rec) const;

    // Recursively builds the subtree for triIndices_[first, last)
    int build(int first, int last);
  };
}