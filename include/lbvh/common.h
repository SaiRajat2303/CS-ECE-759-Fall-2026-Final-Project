// common.h -- shared types, tuning knobs and small helpers.
//
// Everything in this header compiles both as CUDA device code (nvcc) and as
// plain C++ (g++), so the same per-thread logic can be run on the GPU and
// unit-tested on a CPU.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define LBVH_HD __host__ __device__
#define LBVH_INLINE __forceinline__
#else
#define LBVH_HD
#define LBVH_INLINE inline
#endif

// ---------------------------------------------------------------------------
// Compile-time knobs (set with -D on the compiler command line; see Makefile)
// ---------------------------------------------------------------------------

// Threads per block for our own kernels (Morton, hierarchy, refit).
// Also passed to __launch_bounds__ so the compiler can fit registers to it.
#ifndef LBVH_BLOCK_SIZE
#define LBVH_BLOCK_SIZE 256
#endif

// Node storage layout: 0 = array of structures, 1 = structure of arrays.
#ifndef LBVH_LAYOUT_SOA
#define LBVH_LAYOUT_SOA 0
#endif

// Cache hint for read-once ("streaming") loads in the Morton and refit
// kernels: 0 = default (cache in L1 and L2), 1 = ld.global.cg (L2 only).
// The loads that are needed for correctness in refit always bypass L1 and are
// NOT controlled by this knob.
#ifndef LBVH_STREAM_HINT_CG
#define LBVH_STREAM_HINT_CG 0
#endif

// Register cap applied to kernels.cu with -maxrregcount (0 = compiler default).
// The macro only records the value for the CSV output; the flag does the work.
#ifndef LBVH_REG_CAP
#define LBVH_REG_CAP 0
#endif

namespace lbvh {

inline const char* layoutName() { return LBVH_LAYOUT_SOA ? "soa" : "aos"; }
inline const char* streamHintName() { return LBVH_STREAM_HINT_CG ? "cg" : "default"; }

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

struct Vec3 {
  float x, y, z;
};

struct AABB {
  Vec3 lo, hi;
};

LBVH_HD LBVH_INLINE AABB emptyBox() {
  const float inf = INFINITY;
  return AABB{{inf, inf, inf}, {-inf, -inf, -inf}};
}

LBVH_HD LBVH_INLINE AABB pointBox(const Vec3& p) { return AABB{p, p}; }

LBVH_HD LBVH_INLINE AABB merge(const AABB& a, const AABB& b) {
  AABB r;
  r.lo.x = fminf(a.lo.x, b.lo.x);
  r.lo.y = fminf(a.lo.y, b.lo.y);
  r.lo.z = fminf(a.lo.z, b.lo.z);
  r.hi.x = fmaxf(a.hi.x, b.hi.x);
  r.hi.y = fmaxf(a.hi.y, b.hi.y);
  r.hi.z = fmaxf(a.hi.z, b.hi.z);
  return r;
}

// ---------------------------------------------------------------------------
// Loads with an explicit cache policy
// ---------------------------------------------------------------------------

// Load a value that another thread, possibly on another SM, may have just
// written. L1 caches are private to an SM and are not kept coherent with each
// other, so the load must bypass L1 and read from L2 (ld.global.cg).
template <typename T>
LBVH_HD LBVH_INLINE T loadShared(const T* p) {
#if defined(__CUDA_ARCH__)
  return __ldcg(p);
#else
  return *p;
#endif
}

// Load a value that is read once and not reused. Policy set by the
// LBVH_STREAM_HINT_CG knob.
template <typename T>
LBVH_HD LBVH_INLINE T loadStream(const T* p) {
#if defined(__CUDA_ARCH__) && LBVH_STREAM_HINT_CG
  return __ldcg(p);
#else
  return *p;
#endif
}

LBVH_HD LBVH_INLINE Vec3 loadStream(const Vec3* p) {
  Vec3 v;
  v.x = loadStream(&p->x);
  v.y = loadStream(&p->y);
  v.z = loadStream(&p->z);
  return v;
}

// ---------------------------------------------------------------------------
// Morton codes (30 bits: 10 bits per axis)
// ---------------------------------------------------------------------------

constexpr int kMortonBits = 30;

// Spread the low 10 bits of v so that there are two zero bits between each.
LBVH_HD LBVH_INLINE uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Map one coordinate to an integer grid cell in [0, 1023].
// Uses only subtract, divide and multiply, which are exactly rounded on both
// CPU and GPU, so host and device produce identical codes.
LBVH_HD LBVH_INLINE uint32_t quantize(float v, float lo, float hi) {
  const float extent = hi - lo;
  float t = extent > 0.0f ? (v - lo) / extent : 0.0f;
  t = fminf(fmaxf(t * 1024.0f, 0.0f), 1023.0f);
  return static_cast<uint32_t>(t);
}

LBVH_HD LBVH_INLINE uint32_t mortonCode(const Vec3& p, const AABB& scene) {
  const uint32_t x = expandBits(quantize(p.x, scene.lo.x, scene.hi.x));
  const uint32_t y = expandBits(quantize(p.y, scene.lo.y, scene.hi.y));
  const uint32_t z = expandBits(quantize(p.z, scene.lo.z, scene.hi.z));
  return (x << 2) | (y << 1) | z;
}

}  // namespace lbvh
