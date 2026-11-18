#include "bvh.h"
#include "mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace Shader {
  using namespace Mesh;  

  const Vec3 kLightDir = normalize(Vec3(-1.0f, 1.2f, 0.6f));

  Vec3 trace(const Ray& ray, const BVH& bvh, const Mesh::Mesh& mesh, int depth){ 
    HitRecord rec; 
    if(!bvh.intersect(ray, 1e-4f, 1e30f, rec)){ 
      // sky gradient
      float t = 0.5f * (normalize(ray.dir).y + 1.0f);  
      return Vec3(1,1,1) * (1.0f-t) + Vec3(0.5f, 0.7f, 1.0f) * t; 
    }

    const auto& tri = mesh.tris[rec.tri];
    const Vec3& p0 = mesh.positions[tri[0]];
    const Vec3& p1 = mesh.positions[tri[1]];
    const Vec3& p2 = mesh.positions[tri[2]];
    Vec3 hitPoint = ray.at(rec.t);

    // Geometric normal and interpolated shading
    Vec3 ng = normalize(cross(p1-p0, p2-p0));
    float w = 1.0f - rec.u - rec.v;
    const Vec3* cn = &mesh.cornerNormals[rec.tri * 3];
    Vec3 n = normalize(cn[0]*w + cn[1] * rec.u + cn[2] * rec.v); 


    // Make both normals face the viewer (handles meshes with inconsistent winding)
    if(dot(ng, ray.dir) > 0) ng = -ng; 
    if(dot(n, ng) < 0) n = -n;

    // Offset along the geometric normal to avoid self-intersection
    Vec3 offsetPoint = hitPoint + ng * 1e-3f;

    bool inShadow = bvh.occluded(Ray(offsetPoint, kLightDir), 1e-4f, 1e30f);
    float diffuse = inShadow ? 0.0f : std::max(0.0f, dot(n, kLightDir)); 
    Vec3 halfVec = normalize(kLightDir - ray.dir);
    float spec = inShadow? 0.0f: std::pow(std::max(0.0f, dot(n, halfVec)), 64.0f);

    Vec3 base = mesh.triColors[rec.tri]; 
    Vec3 color = base * (0.15f + 0.85f * diffuse) + Vec3(1,1,1) * (0.4f * spec);

    if (depth > 0){ // one bounce off mirror reflection
      Vec3 r = ray.dir - n * (2.0f * dot(ray.dir, n)); 
      color = color * 0.85f + trace(Ray(offsetPoint, normalize(r)), bvh, mesh, depth - 1) * 0.15f;
    }

    return color; 
  }

  bool loadOBJ(const std::string& path, Mesh::Mesh& out, const Vec3& color){ 
    std::ifstream in(path); 
    if(!in) return false; 
    std::string line; 
    while(std::getline(in, line)){ 
      std::istringstream ss(line);
      std::string tag; 
      ss >> tag; 
      if(tag == "v"){ 
        Vec3 p; ss >> p.x >> p.y >> p.z;
        out.positions.push_back(p);
      } else if (tag == "f"){ 
        std::vector<int> idx; 
        std::string tok; 
        while(ss >> tok){ 
          int i = std::stoi(tok.substr(0, tok.find('/'))); 
          idx.push_back(i > 0 ? i - 1 : (int)out.positions.size());
        }
        for(size_t k = 1; k +1 < idx.size(); ++k){ 
          out.tris.push_back({idx[0], idx[k], idx[k+1]});
          out.triColors.push_back(color);
        }
      }
    }
    out.computeNormals(); 
    return !out.tris.empty();
  }

  enum class Mode { Rigid, Twist, Wave, Explode }; 

  inline Mode parseMode(const std::string& s){ 
    if(s == "rigid") return Mode::Rigid;
    if(s == "twist") return Mode::Twist;
    if(s == "wave") return Mode::Wave;
    if(s == "explode") return Mode::Explode;
    throw std::runtime_error("unknown mode: " + s + "(rigid|twist|wave|explode)");
  }

  // produces vertex position for t in [0,1]. Call fram (i, n) for frame i of n
  // class Animator {
  //   Animator(const Mesh& rest, Mode mode, float amount=1.0f, uint32_t seed = 42) : mode_(mode), amount_(amount) { 
  //     mesh_ = (mode == Mode::Explode)? rest : nullptr;
  //   }


  //   private: 
  //     Mesh mesh_; 
  //     Mode mode_; 
  //     float amount_;
  //     std::vector<Vec3> triDir_; 
  //     std::vector<Vec3> triSpin_; 
  // }



  Mesh::Mesh makeTorus(const Vec3& center, float R, float r, int segU, int segV, const Vec3& color){ 
    Mesh::Mesh m;
    const float kTwoPi = 6.28318530718f;
    for (int i = 0; i < segU; ++i) {
        float a = kTwoPi * i / segU;
        for (int j = 0; j < segV; ++j) {
            float b = kTwoPi * j / segV;
            // Torus lying in the XZ plane, then tilted for a more interesting view.
            Vec3 p((R + r * std::cos(b)) * std::cos(a), r * std::sin(b), (R + r * std::cos(b)) * std::sin(a));
            float tilt = 0.5f;
            Vec3 q(p.x, p.y * std::cos(tilt) - p.z * std::sin(tilt), p.y * std::sin(tilt) + p.z * std::cos(tilt));
            m.positions.push_back(center + q);
        }
    }
    for (int i = 0; i < segU; ++i)
        for (int j = 0; j < segV; ++j) {
            int i1 = (i + 1) % segU, j1 = (j + 1) % segV;
            int a = i * segV + j, b = i1 * segV + j, c = i1 * segV + j1, d = i * segV + j1;
            m.tris.push_back({a, d, c});
            m.tris.push_back({a, c, b});
            m.triColors.push_back(color);
            m.triColors.push_back(color);
        }
    m.computeNormals();
    return m;
  }
  //An icosahedron has 12 vertices, 30 edges and 20 faces. Every face is an identical equilateral triangle,
  // and every vertex has the same five neighbors. It’s the most “sphere-like” shape you can make out of
  // identical equilateral triangles: no point on it is special, so it makes an evenly balanced starting point.
  Mesh::Mesh makeIcosphere(const Vec3& center, float radius, int subdivisions, const Vec3& color){
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<Vec3> v = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
        {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    for (auto& p : v) p = normalize(p);
    std::vector<std::array<int, 3>> f = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
        {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
        {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};

    for (int s = 0; s < subdivisions; ++s) {
        std::map<std::pair<int, int>, int> midCache;  // share midpoints between neighbors
        auto midpoint = [&](int a, int b) {
            auto key = std::minmax(a, b);
            auto it = midCache.find(key);
            if (it != midCache.end()) return it->second;
            v.push_back(normalize((v[a] + v[b]) * 0.5f));
            return midCache[key] = (int)v.size() - 1;
        };
        std::vector<std::array<int, 3>> nf;
        for (auto& tri : f) {
            int a = midpoint(tri[0], tri[1]), b = midpoint(tri[1], tri[2]), c = midpoint(tri[2], tri[0]);
            nf.push_back({tri[0], a, c});
            nf.push_back({tri[1], b, a});
            nf.push_back({tri[2], c, b});
            nf.push_back({a, b, c});
        }
        f.swap(nf);
    }

    Mesh::Mesh m;
    for (auto& p : v) m.positions.push_back(center + p * radius);
    m.tris = f;
    m.triColors.assign(f.size(), color);
    m.computeNormals();
    return m;

  }

} // namespace Shader

int main(int argc, char** argv){ 
    using namespace Shader;
    const int width = 800, height = 459; 

    Mesh::Mesh scene; 
    scene.append(makeQuad(Vec3(-50,0,-50),Vec3(-50,0,50),Vec3(50,0,50), Vec3(50,0,-50), Vec3(0.55f, 0.55f, 0.55f)));

    if(argc > 1){ 
      Mesh::Mesh model; 
      if(!loadOBJ(argv[1], model, Vec3(0.8f, 0.6f, 0.4f))){
        std::cerr << "Failed to load OBJ: " << argv[1] << "\n"; 
        return 1;
      }
      fitToGround(model, 3.0f); 
      scene.append(model);
    } else { 
      scene.append(makeTorus(Vec3(-1.6f, 1.6f, 0.0f), 1.2f, 0.45f, 240, 80, Vec3(0.9f, 0.35f, 0.2f)));
      scene.append(makeIcosphere(Vec3(1.8f, 1.2f, 0.5f), 1.2f, 4, Vec3(0.25f, 0.5f, 0.9f)));
    }

    BVH bvh(scene);
    std::cerr << "Built BVH: " << scene.tris.size() << " triangles, "
          << bvh.nodeCount() << " nodes\n"; 

    // Pinhold camera; 
    Vec3 eye(0,3.5f, 9), target(0,1.3f,0), up(0,1,0);
    float fov = 40.0f * 3.14159265f / 180.0f;
    Vec3 wv = normalize(eye - target);
    Vec3 uv = normalize(cross(up, wv)); 
    Vec3 vv = cross(wv, uv);
    float halfH = std::tan(fov / 2); 
    float halfW = halfH * (float)width / height;



    
    // Output PPM image to stdout
    auto toByte = [](float c) {
      return (int)(255.99f * std::sqrt(std::min(1.0f, std::max(0.0f, c))));  // gamma 2
    };

    std::cout << "P3\n" << width << ' ' << height << "\n255\n";
    for(int j = 0; j < height; ++j){ 
      for(int i = 0; i < width; ++i){ 
        float s = (2.0f * (i + 0.5f) / width - 1.0f) * halfW;
        float t = (1.0f - 2.0f * (j + 0.5f) / height ) * halfH; 
        Vec3 dir = normalize(uv * s + vv * t - wv);
        Vec3 col = trace(Ray(eye, dir), bvh, scene, 1);
        std::cout << toByte(col.x) << ' ' << toByte(col.y) << ' ' << toByte(col.z) << '\n';
      }
    }

    std::cerr << "Done.\n";
    return 0;
  }
