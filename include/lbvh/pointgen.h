// pointgen.h -- synthetic point clouds and a simple .xyz loader.
// The generators use their own random number code, so the same seed gives the
// same points regardless of the C++ standard library in use.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "lbvh/common.h"

namespace lbvh {

enum class Dist { Uniform, Gaussian, Clusters, Surface, Duplicates };

inline const char* distName(Dist d) {
  switch (d) {
    case Dist::Uniform: return "uniform";
    case Dist::Gaussian: return "gaussian";
    case Dist::Clusters: return "clusters";
    case Dist::Surface: return "surface";
    case Dist::Duplicates: return "duplicates";
  }
  return "?";
}

inline bool parseDist(const std::string& s, Dist& d) {
  if (s == "uniform") d = Dist::Uniform;
  else if (s == "gaussian") d = Dist::Gaussian;
  else if (s == "clusters") d = Dist::Clusters;
  else if (s == "surface") d = Dist::Surface;
  else if (s == "duplicates") d = Dist::Duplicates;
  else return false;
  return true;
}

// splitmix64: small and fast.
struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed) {}
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [0, 1).
  float uniform() { return static_cast<float>(next() >> 40) * (1.0f / 16777216.0f); }
  // Standard normal (Box-Muller).
  float normal() {
    const double u1 = (static_cast<double>(next() >> 11) + 1.0) * (1.0 / 9007199254740993.0);
    const double u2 = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
};

inline std::vector<Vec3> generatePoints(Dist dist, int n, uint64_t seed) {
  Rng rng(seed * 0x2545F4914F6CDD1Dull + 0x1234567ull);
  std::vector<Vec3> pts(n);
  switch (dist) {
    case Dist::Uniform:  // unit cube
      for (auto& p : pts) p = {rng.uniform(), rng.uniform(), rng.uniform()};
      break;
    case Dist::Gaussian:  // one blob
      for (auto& p : pts)
        p = {0.5f + 0.15f * rng.normal(), 0.5f + 0.15f * rng.normal(), 0.5f + 0.15f * rng.normal()};
      break;
    case Dist::Clusters: {  // 16 tight blobs
      Vec3 centers[16];
      for (auto& c : centers) c = {rng.uniform(), rng.uniform(), rng.uniform()};
      for (auto& p : pts) {
        const Vec3& c = centers[rng.next() % 16];
        p = {c.x + 0.01f * rng.normal(), c.y + 0.01f * rng.normal(), c.z + 0.01f * rng.normal()};
      }
      break;
    }
    case Dist::Surface:  // points on a sphere, like a scanned object
      for (auto& p : pts) {
        float x = rng.normal(), y = rng.normal(), z = rng.normal();
        float len = std::sqrt(x * x + y * y + z * z);
        if (len == 0.0f) { x = 1.0f; len = 1.0f; }
        p = {0.5f + 0.5f * x / len, 0.5f + 0.5f * y / len, 0.5f + 0.5f * z / len};
      }
      break;
    case Dist::Duplicates: {  // n/16 distinct positions, each repeated ~16 times
      const int distinct = n / 16 > 0 ? n / 16 : 1;
      std::vector<Vec3> base(distinct);
      for (auto& b : base) b = {rng.uniform(), rng.uniform(), rng.uniform()};
      for (auto& p : pts) p = base[rng.next() % distinct];
      break;
    }
  }
  return pts;
}

// Reads an ASCII file with one "x y z" triple per line (extra columns and
// lines that do not start with three numbers are ignored).
inline bool loadXYZ(const std::string& path, std::vector<Vec3>& pts) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return false;
  char line[512];
  pts.clear();
  while (std::fgets(line, sizeof line, f)) {
    Vec3 p;
    if (std::sscanf(line, "%f %f %f", &p.x, &p.y, &p.z) == 3) pts.push_back(p);
  }
  std::fclose(f);
  return !pts.empty();
}

}  // namespace lbvh
