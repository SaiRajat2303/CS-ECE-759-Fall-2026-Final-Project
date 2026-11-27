// dynamic_scene.h — turns a static triangle mesh into an animated sequence
// with constant topology (same vertex count and triangle list every frame).
//
// Modes, for frame time t in [0, 1] and strength `amount`:
//   rigid    the whole mesh rotates about the vertical axis and drifts sideways.
//            Triangles keep their shape and neighbours, but axis-aligned
//            boxes of rotated subtrees grow, so refitted trees get looser.
//   twist    rotation about the vertical axis by an angle that grows with
//            height: neighbouring triangles move coherently.
//   wave     a travelling plane wave displaces vertices vertically.
//   explode  every triangle flies outward from the centre along its own
//            direction and spins: the worst case for refitting, because
//            triangles that were close at frame 0 end up far apart.
//
// Header-only, host-only, no dependencies beyond the standard library.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dyn {

struct Vec3 {
  float x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

inline float length(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
inline Vec3 normalize(const Vec3& v) {
  const float l = length(v);
  return l > 0 ? v * (1.0f / l) : Vec3(0, 1, 0);
}

// Rotate p about the unit axis k by angle a (Rodrigues).
inline Vec3 rotate(const Vec3& p, const Vec3& k, float a) {
  const float c = std::cos(a), s = std::sin(a);
  const float d = k.x * p.x + k.y * p.y + k.z * p.z;
  const Vec3 kxp(k.y * p.z - k.z * p.y, k.z * p.x - k.x * p.z, k.x * p.y - k.y * p.x);
  return p * c + kxp * s + k * (d * (1 - c));
}

using Tri = std::array<uint32_t, 3>;

struct Mesh {
  std::vector<Vec3> verts;
  std::vector<Tri> tris;

  // Centre the mesh at the origin and scale its largest extent to 1.
  void normalizeToUnitBox() {
    if (verts.empty()) return;
    Vec3 lo = verts[0], hi = verts[0];
    for (const auto& v : verts) {
      lo = {std::fmin(lo.x, v.x), std::fmin(lo.y, v.y), std::fmin(lo.z, v.z)};
      hi = {std::fmax(hi.x, v.x), std::fmax(hi.y, v.y), std::fmax(hi.z, v.z)};
    }
    const Vec3 c = (lo + hi) * 0.5f;
    const float ext = std::fmax(hi.x - lo.x, std::fmax(hi.y - lo.y, hi.z - lo.z));
    const float s = ext > 0 ? 1.0f / ext : 1.0f;
    for (auto& v : verts) v = (v - c) * s;
  }
};

enum class Mode { Rigid, Twist, Wave, Explode };

inline Mode parseMode(const std::string& s) {
  if (s == "rigid") return Mode::Rigid;
  if (s == "twist") return Mode::Twist;
  if (s == "wave") return Mode::Wave;
  if (s == "explode") return Mode::Explode;
  throw std::runtime_error("unknown mode: " + s + " (rigid|twist|wave|explode)");
}

class Animator {
 public:
  // For Explode the triangles stop sharing vertices (each gets its own three),
  // so they can separate; restVerts()/tris() describe that unshared mesh.
  Animator(const Mesh& rest, Mode mode, float amount = 1.0f, uint32_t seed = 42)
      : mode_(mode), amount_(amount) {
    if (mode != Mode::Explode) {
      verts_ = rest.verts;
      tris_ = rest.tris;
      return;
    }
    verts_.reserve(rest.tris.size() * 3);
    tris_.reserve(rest.tris.size());
    triOf_.reserve(rest.tris.size() * 3);
    centroid_.reserve(rest.tris.size());
    dir_.reserve(rest.tris.size());
    spinAxis_.reserve(rest.tris.size());
    speed_.reserve(rest.tris.size());
    uint32_t rng = seed;
    for (uint32_t t = 0; t < rest.tris.size(); ++t) {
      const Tri& tri = rest.tris[t];
      const uint32_t base = static_cast<uint32_t>(verts_.size());
      Vec3 c;
      for (int k = 0; k < 3; ++k) {
        verts_.push_back(rest.verts[tri[k]]);
        triOf_.push_back(t);
        c = c + rest.verts[tri[k]];
      }
      c = c * (1.0f / 3.0f);
      tris_.push_back({base, base + 1, base + 2});
      centroid_.push_back(c);
      const Vec3 jitter(uniform(rng) - 0.5f, uniform(rng) - 0.5f, uniform(rng) - 0.5f);
      dir_.push_back(normalize(normalize(c) + jitter * 0.6f));
      spinAxis_.push_back(normalize(Vec3(uniform(rng) - 0.5f, uniform(rng) - 0.5f, uniform(rng) - 0.5f)));
      speed_.push_back(0.5f + uniform(rng));
    }
  }

  const std::vector<Vec3>& restVerts() const { return verts_; }
  const std::vector<Tri>& tris() const { return tris_; }

  // Vertex positions of frame i of n (t = i / (n - 1)).
  std::vector<Vec3> frame(int i, int n) const {
    const float t = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
    const float kPi = 3.14159265358979f;
    std::vector<Vec3> out(verts_.size());
    switch (mode_) {
      case Mode::Rigid: {
        const float a = kPi * t * amount_;  // half a turn over the sequence
        const Vec3 shift(0.5f * amount_ * t, 0.0f, 0.0f);
        for (size_t k = 0; k < verts_.size(); ++k) out[k] = rotate(verts_[k], Vec3(0, 1, 0), a) + shift;
        break;
      }
      case Mode::Twist: {
        const float maxAngle = kPi * amount_ * t;  // radians per unit height
        for (size_t k = 0; k < verts_.size(); ++k) {
          const Vec3& v = verts_[k];
          out[k] = rotate(v, Vec3(0, 1, 0), maxAngle * (v.y + 0.5f));
        }
        break;
      }
      case Mode::Wave: {
        const float amp = 0.08f * amount_;
        // Plane wave travelling along (3, 2) cycles per unit; it grows in from
        // zero so frame 0 is the rest pose, and moves one wavelength in total.
        const float phase = 2.0f * kPi * t;
        const float grow = std::fmin(1.0f, 4.0f * t);
        for (size_t k = 0; k < verts_.size(); ++k) {
          const Vec3& v = verts_[k];
          out[k] = {v.x, v.y + grow * amp * std::sin(2.0f * kPi * (3.0f * v.x + 2.0f * v.z) - phase), v.z};
        }
        break;
      }
      case Mode::Explode: {
        const float dist = 0.6f * amount_ * t;
        const float spin = 4.0f * kPi * amount_ * t;
        for (size_t k = 0; k < verts_.size(); ++k) {
          const uint32_t tri = triOf_[k];
          const Vec3 c = centroid_[tri];
          const Vec3 local = rotate(verts_[k] - c, spinAxis_[tri], spin * speed_[tri]);
          out[k] = c + dir_[tri] * (dist * speed_[tri]) + local;
        }
        break;
      }
    }
    return out;
  }

 private:
  static float uniform(uint32_t& s) {  // xorshift32 in [0, 1)
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >> 8) * (1.0f / 16777216.0f);
  }

  Mode mode_;
  float amount_;
  std::vector<Vec3> verts_;
  std::vector<Tri> tris_;
  // Explode only, per vertex / per triangle:
  std::vector<uint32_t> triOf_;
  std::vector<Vec3> centroid_, dir_, spinAxis_;
  std::vector<float> speed_;
};

}  // namespace dyn
