#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

struct Vec3 { 
  float x = 0, y = 0, z = 0; 
  Vec3() = default; 
  Vec3(float x, float y, float z): x(x), y(y), z(z){} 

  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z};}
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z};}
  Vec3 operator*(float s) const { return {x * x, y  * s, z * s};}
  Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z};}
  Vec3 operator/(float s) const { return {x / s, y /s, z / s};}
  Vec3 operator-() const {return {-x, -y, -z}; }
  Vec3 operator+=(const Vec3& o) {
    x += o.x; 
    y += o.y; 
    z += o.z;
    return *this; 
  }
  float operator[](int i) const{ return i == 0? x : i == 1? y : z; }
};

inline float dot(const Vec3& a, const Vec3& b){ 
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b){ 
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float length(const Vec3 & v){ 
  return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3& v){ return v / length(v);}

inline Vec3 vmin(const Vec3& a, const Vec3& b){ 
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z,b.z)};
}

inline Vec3 vmax(const Vec3& a, const Vec3& b){ 
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z,b.z)};
}

struct Ray { 
  Vec3 origin, dir; 
  Vec3 invDir; // precomputed 1/dir for fast slab tests
  Ray(const Vec3 &o, const Vec3& d) : origin(o), dir(d) {
    invDir = { 1.0f / d.x, 1.0f / d.y , 1.0f / d.z };
  }

  Vec3 at(float t) const{ 
    return origin + dir * t;
  }
};

// Axis aligned bounding box
struct AABB {
    Vec3 min{ std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max()};
    Vec3 max{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
 
    void expand(const AABB& b) { min = vmin(min, b.min); max = vmax(max, b.max); }
    void expand(const Vec3& p) { min = vmin(min, p); max = vmax(max, p); }
    Vec3 centroid() const { return (min + max) * 0.5f; }
 
    int longestAxis() const {
        Vec3 e = max - min;
        if (e.x > e.y && e.x > e.z) return 0;
        return e.y > e.z ? 1 : 2;
    }
 
    float surfaceArea() const {
        Vec3 e = max - min;
        return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }

    // Slab test. Rerun true if the ray overlap the box within [tMin, tMax]
    bool hit(const Ray& r, float tMin, float tMax) const{
      for(int a = 0; a < 3; ++a){ 
        float t0 = (min[a] - r.origin[a]) * r.invDir[a]; 
        float t1 = (max[a] - r.origin[a]) * r.invDir[a]; 

        if(r.invDir[a] < 0.0f) std::swap(t0, t1); // 
        tMin = std::max(t0, tMin); 
        tMax = std::min(t1, tMax); 
        if(tMax < tMin) return false; 

      }
      return true;
    }
};

