#pragma once
#ifndef MESH_H
#define MESH_H

#include <array>
#include <vector>

#include "common.h"

namespace Mesh {
  struct Triangle { 
    Vec3 x, y, z; 
    Triangle() = default; 
    Triangle(const Vec3& x, const Vec3& y, const Vec3& z): x(x) , y(y), z(z) {}
  };

  struct Mesh
  {
    std::vector<Vec3> positions;
    std::vector<Vec3> cornerNormals; 
    std::vector<std::array<int,3>> tris; // indices into normals 
    std::vector<Vec3> triColors;  // one color per triangle

    // Append another mesh, offsetting its indices.
    void append(const Mesh& m){ 
      int base = (int)positions.size();
      positions.insert(positions.end(), m.positions.begin(), m.positions.end());
      cornerNormals.insert(cornerNormals.end(), m.cornerNormals.begin(), m.cornerNormals.end());
      
      for(auto t : m.tris) tris.push_back({t[0] + base, t[1] + base, t[2]+ base});
      triColors.insert(triColors.end(), m.triColors.begin(), m.triColors.end());
    }


    // Compute the smooth normal with a crease angle. At each triangle corner the
    // normal averages the faces that share the vertex whose normals are within 
    // creaseDeg of this face
    void computeNormals(float creaseDeg = 45.0f) { 
      size_t nt = tris.size(); 
      std::vector<Vec3> faceN(nt), faceUnit(nt); 
      std::vector<std::vector<int>> vertFaces(positions.size());
      for (size_t f = 0; f < nt; ++f) {
          const auto& t = tris[f];
          faceN[f] = cross(positions[t[1]] - positions[t[0]], positions[t[2]] - positions[t[0]]);
          faceUnit[f] = normalize(faceN[f]);
          for (int k = 0; k < 3; ++k) vertFaces[t[k]].push_back((int)f);
      }
      float cosCrease = std::cos(creaseDeg * 3.14159265f / 180.0f);
      cornerNormals.assign(nt * 3, Vec3()); 
      for(size_t f = 0; f < nt; ++f){ 
        for(int k = 0; k < 3; ++k){ 
          Vec3 sum; 
          for(int g : vertFaces[tris[f][k]]) {
            if (dot(faceUnit[f], faceUnit[g]) >= cosCrease) sum += faceN[g];
          }
          cornerNormals[f * 3 + k] = sum;
        }
      }
    }

    AABB triBounds(int i) const { 
      AABB b; 
      for (int k = 0; k < 3; ++k){ 
        b.expand(positions[tris[i][k]]);
      }
      return b;
    }

    // Give each triangle its own 3 vertices (needed for Explode)
    Mesh unwelded() const { 
      Mesh m; 
      for (const auto& t: tris){
        int b = (int)m.positions.size(); 
        for (int k = 0; k < 3; ++k) m.positions.push_back(positions[t[k]]);
        m.tris.push_back({b, b+1, b+2});
      }
      m.triColors = triColors;
      m.cornerNormals = cornerNormals; // corner normals are already per (tri, corner)
      return m;
    }

  };

  struct HitRecord { 
    float t = std::numeric_limits<float>::max(); 
    int tri = -1; 
    float u = 0, v = 0; // barycentric coordinates of hit
  };

  // Möller–Trumbore ray/triangle intersection.
  // Solves  origin + t*dir = (1-u-v)*v0 + u*v1 + v*v2  for (t, u, v).
  inline bool intersectTriangle(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, float tMin, float tMax, float& t, float& u, float& v){ 
    const float kEps = 1e-8f; 
    Vec3 e1 = v1 - v0; 
    Vec3 e2 = v2 - v0; 
    Vec3 p = cross(ray.dir, e2);
    float det = dot(e1, p);
    if(std::fabs(det) < kEps) return false; // ray parallel to triangle plane
    float invDet = 1.0f/det; 

    Vec3 s = ray.origin - v0; 
    u = dot(s, p) * invDet;
    if(u < 0.0f || u > 1.0f) return false;  

    Vec3 q = cross(s, e1); 
    v = dot(ray.dir, q) * invDet;
    if(v < 0.0f || u + v > 1.0f) return false; 

    t = dot(e2, q) * invDet; 
    return t > tMin && t < tMax;
  }

  // Uniformly scal and move a mesh so it sits on the ground (y=0) 
  inline void fitToGround(Mesh& m, float targetHeight){ 
    AABB b; 
    for(auto& p : m.positions) b.expand(p); 
    Vec3 ext = b.max - b.min; 
    float s = targetHeight / std::max(ext.y, 1e-6f); 
    Vec3 c = b.centroid(); 
    // center in x/z, put the lowest point at y = 0
    for(auto& p : m.positions) p = Vec3((p.x - c.x) * s, (p.y - b.min.y) * s, (p.z - c.z) * s);
  }

  // Mesh generators
  inline Mesh makeQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, const Vec3& color){ 
    Mesh m;
    m.positions = {a,b,c,d}; 
    m.tris = {{0,1,2}, {0,2,3}}; 
    m.triColors = {color, color}; 
    m.computeNormals(); 
    return m;
  }
}
#endif