#include "bvh.h"

#include <algorithm>
#include <limits>

namespace Shader 
{ 
  BVH::BVH(const Mesh::Mesh& mesh) : mesh_(mesh){
    size_t n = mesh.tris.size(); 
    triIndices_.resize(n);
    triBounds_.resize(n); 
    centroids_.resize(n); 
    for(size_t i = 0; i < n; ++i){ 
      triIndices_[i] = (int)i; 
      triBounds_[i] = mesh.triBounds((int)i);
      centroids_[i] = triBounds_[i].centroid();
    }
    nodes_.reserve(n*2); 
    if (n > 0) build(0, int(n));
  }

  bool BVH::intersect(const Ray& ray, float tMin, float tMax, Mesh::HitRecord& rec) const { 
    return traverse(ray, tMin, tMax, &rec);
  }

  // Any-hit query: stops at the first hit
  bool BVH::occluded(const Ray& ray, float tMin, float tMax) const { 
    return traverse(ray, tMin, tMax, nullptr);
  }

  bool BVH::traverse(const Ray& ray, float tMin, float tMax, Mesh::HitRecord* rec) const { 
    bool hitAnything = false; 
    float closest = tMax; 
    
    if (nodes_.empty() || mesh_.tris.empty()) return false;

    int stack[64]; 
    int sp = 0; 
    stack[sp++] = 0; // root

    while (sp > 0){ 
      const BVHNode& node = nodes_[stack[--sp]];
      if(!node.bounds.hit(ray, tMin, closest)) continue; 

      if(node.isLeaf()) { 
        for(int i = 0; i < node.count; ++i){ 
          int ti = triIndices_[node.first + i]; 
          const auto& tri = mesh_.tris[ti]; 
          float t, u, v; 
          if(Mesh::intersectTriangle(ray, mesh_.positions[tri[0]], mesh_.positions[tri[1]], mesh_.positions[tri[2]], tMin, closest, t, u, v)){ 
            if (!rec) return true; 
            hitAnything = true;
            closest = t; // shrink the interval; prunes farther boxes
            rec->t = t; rec->tri = ti; rec->u = u; rec->v = v; 
          }
        }
      } else { 
        // Visit the nearest chilf first so close hits are found early
        const BVHNode& Lhs = nodes_[node.left]; 
        const BVHNode& Rhs = nodes_[node.right]; 
        
        float dl = dot(Lhs.bounds.centroid() - ray.origin, ray.dir);
        float dr = dot(Rhs.bounds.centroid() - ray.origin, ray.dir);
        if(dl < dr){
          stack[sp++] = node.right;
          stack[sp++] = node.left; 
        } else {
          stack[sp++] = node.left;
          stack[sp++] = node.right;
        }
      }

    }
    return hitAnything;
  }

  // Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
  // Each multiply copies the value shifted left (v * (1 + 2^k) == v + (v << k)),
  // and the mask keeps only the bits that landed in the right slots.
  static unsigned int expandBits(unsigned int v){ 
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
  }
  
  // Calculate 30-bit morton code for the given 3D point 
  // in the unit cube [0, 1]
  [[maybe_unused]] static unsigned int morton3D(float x, float y, float z){ 
    // expand bits to 10
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f); 
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x); 
    unsigned int yy = expandBits((unsigned int)y); 
    unsigned int zz = expandBits((unsigned int)z); 

    return xx * 4 + yy * 2 + zz; 
  }


  // Recursively builds the subtree for triInidices_[first, last).
  int BVH::build(int first, int last){ 
    int nodeIdx = (int)nodes_.size(); 
    nodes_.emplace_back(); 

    AABB bounds, centroidBounds; 
    for (int i = first; i < last; i++){ 
      bounds.expand(triBounds_[triIndices_[i]]);
      centroidBounds.expand(centroids_[triIndices_[i]]);
    }

    nodes_[nodeIdx].bounds = bounds; 
    int n = last - first; 
    auto makeLeaf = [&](){ 
      nodes_[nodeIdx].first = first; 
      nodes_[nodeIdx].count = n; 
      return nodeIdx; 
    };

    if (n <= kMaxLeafSize) return makeLeaf();

    int axis = centroidBounds.longestAxis(); 
    float cMin = centroidBounds.min[axis]; 
    float cMax = centroidBounds.max[axis];
    if(cMax - cMin < 1e-9f) return makeLeaf(); // centroids coincide


    // Binned Surface Area Heurists
    // cost(split) = 1 + (SA(left) * Nleft + SA(right)*Nright) / SA(parent)
    struct Bucket {int count = 0; AABB bounds;};
    Bucket buckets[kSahBuckets];
    auto bucketOf = [&](int tri){ 
      int b = (int)(kSahBuckets * (centroids_[tri][axis] - cMin) / (cMax - cMin));
      return std::min(b, kSahBuckets - 1);
    }; 
    for(int i = first; i < last; ++i){ 
      int t = triIndices_[i]; 
      Bucket& b= buckets[bucketOf(t)]; 
      b.count++; 
      b.bounds.expand(triBounds_[t]);
    }

    float leftArea[kSahBuckets], rightArea[kSahBuckets]; 
    int leftCount[kSahBuckets], rightCount[kSahBuckets]; 

    AABB acc; int cnt = 0; 
    for(int i = 0; i < kSahBuckets; ++i){ 
      acc.expand(buckets[i].bounds); 
      cnt += buckets[i].count;
      leftArea[i] = cnt ? acc.surfaceArea() : 0.0f; 
      leftCount[i] = cnt; 
    }
    acc = AABB(); cnt = 0; 
    for(int i = kSahBuckets -1; i >= 0; --i){ 
      acc.expand(buckets[i].bounds); 
      cnt += buckets[i].count;
      rightArea[i] = cnt ? acc.surfaceArea() : 0.0f; 
      rightCount[i] = cnt; 
    }

    float parentArea = std::max(bounds.surfaceArea(), 1e-12f); 
    float bestCost = std::numeric_limits<float>::max(); 
    int bestSplit = -1;
    for (int s = 0; s < kSahBuckets - 1; ++s){ 
      int nl = leftCount[s], nr = rightCount[s+1]; 
      if (nl == 0 || nr == 0) continue; 
      float cost = 1.0f + (leftArea[s] * nl + rightArea[s+1] * nr) / parentArea; 
      if(cost < bestCost) { bestCost = cost; bestSplit = s;}
    }

    if(bestSplit < 0 || bestCost >= (float)n) return makeLeaf(); 

    int* begin = triIndices_.data() + first;
    int* mid = std::partition(begin, begin + n, [&](int t) { return bucketOf(t) <= bestSplit; });
    int midIdx = (int)(mid - triIndices_.data());
    if(midIdx  == first || midIdx == last) return makeLeaf(); 

    int left = build(first, midIdx); 
    int right = build(midIdx, last);
    nodes_[nodeIdx].left = left; 
    nodes_[nodeIdx].right = right; 
    return nodeIdx;
  }
}