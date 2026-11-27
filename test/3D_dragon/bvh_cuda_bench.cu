// bvh_cuda_bench.cu — per-frame GPU benchmark for a custom CUDA BVH on an
// animated triangle mesh.
//
// Input: a sequence of PLY or OBJ frames with identical topology (same vertex
// count and triangle list, only positions change), e.g. from make_frames
// (Stanford PLY -> animated frames), baked from Blender, or BART / Utah.
//
// Dependencies (fetched by CMake): tinyobjloader, happly — see mesh_io.h
//
// Build:  cmake -B build && cmake --build build -j
// Run:    ./build/bvh_cuda_bench "frames/frame_%04d.ply" 60 results.csv
//         options: --cam-res N   NxN coherent camera rays   (default 1024)
//                  --rnd-rays N  incoherent random rays     (default 1048576)
//                  --repeats R   trace launches averaged    (default 5)
//                  --first F     index of first frame file  (default 0)
//                  --label NAME  run name (e.g. the animation mode): shown in
//                                the NVTX ranges and as the CSV's first column
//
// The UserBVH adapter (lbvh_user_bvh.cuh) plugs in the LBVH from include/lbvh/.
// Everything else (loading, ray generation, timing, correctness check, CSV)
// is done here.

#define MESH_IO_IMPLEMENTATION
#include "mesh_io.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// NVTX ranges label each frame / strategy / phase in the Nsight Systems
// timeline (nsys profile --trace=cuda,nvtx). Header-only; compiled out if the
// toolkit has no NVTX headers.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};
#else
struct NvtxRange { explicit NvtxRange(const char*) {} };
#endif

#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t e_ = (x);                                                      \
        if (e_ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                     \
                         cudaGetErrorString(e_), __FILE__, __LINE__);              \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------- types -----
struct Ray { float3 o; float tmin; float3 d; float tmax; };   // 32 bytes
struct Hit { float t; int prim; float u, v; };                // prim = -1 on miss

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ inline float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 cross3(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__host__ __device__ inline float3 norm3(float3 a) { float l = sqrtf(dot3(a, a)); return l > 0 ? a * (1.0f / l) : a; }

__global__ void fillMissKernel(Hit* hits, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hits[i] = Hit{INFINITY, -1, 0.f, 0.f};
}

// =============================================================================
// ADAPTER — the LBVH from include/lbvh/ (Karras 2012), one leaf per triangle.
// build(): full rebuild; update(): refit with the frame-0 topology.
// See lbvh_user_bvh.cuh and include/lbvh/tri_impl.h.
// =============================================================================
#include "lbvh_user_bvh.cuh"

// ----------------------------------------------------------- ray kernels -----
__device__ inline uint32_t pcgHash(uint32_t v) {
    uint32_t s = v * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
__device__ inline float rnd01(uint32_t& s) { s = pcgHash(s); return s * (1.0f / 4294967296.0f); }

__global__ void genCameraRays(Ray* rays, int w, int h, float3 eye, float3 fwd, float3 right, float3 up, float tanH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float aspect = float(w) / float(h);
    float u = ((x + 0.5f) / w * 2.f - 1.f) * tanH * aspect;
    float v = (1.f - (y + 0.5f) / h * 2.f) * tanH;
    rays[y * w + x] = Ray{eye, 0.f, norm3(fwd + right * u + up * v), INFINITY};
}

// Random origins inside the scene box, random directions: stresses BVH quality.
__global__ void genRandomRays(Ray* rays, int n, float3 center, float halfExtent, uint32_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s = pcgHash(i ^ seed);
    float3 o = make_float3(rnd01(s) * 2 - 1, rnd01(s) * 2 - 1, rnd01(s) * 2 - 1);
    float3 d;
    do { d = make_float3(rnd01(s) * 2 - 1, rnd01(s) * 2 - 1, rnd01(s) * 2 - 1); }
    while (dot3(d, d) > 1.f || dot3(d, d) < 1e-4f);
    rays[i] = Ray{center + o * halfExtent, 0.f, norm3(d), INFINITY};
}

// Correctness check: rebuild vs. update trace the SAME geometry, so their
// closest hits must agree (up to float noise / ties).
__global__ void compareHits(const Hit* a, const Hit* b, int n, float relEps,
                            unsigned int* mismatches, unsigned int* hitsA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool ha = a[i].prim >= 0, hb = b[i].prim >= 0;
    if (ha) atomicAdd(hitsA, 1u);
    bool bad = (ha != hb) || (ha && fabsf(a[i].t - b[i].t) > relEps * fmaxf(1.f, fabsf(a[i].t)));
    if (bad) atomicAdd(mismatches, 1u);
}

// --------------------------------------------------------------- loading -----
static bool loadFrame(const std::string& path, std::vector<float3>& verts, std::vector<int3>* tris) {
    try {
        meshio::TriMesh m = meshio::load(path);   // .ply (happly) or .obj (tinyobjloader)
        verts.resize(m.positions.size());
        for (size_t i = 0; i < verts.size(); ++i)
            verts[i] = make_float3(m.positions[i][0], m.positions[i][1], m.positions[i][2]);
        if (tris) {
            tris->resize(m.triangles.size());
            for (size_t i = 0; i < tris->size(); ++i)
                (*tris)[i] = make_int3(m.triangles[i][0], m.triangles[i][1], m.triangles[i][2]);
        }
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load %s: %s\n", path.c_str(), e.what());
        return false;
    }
}

static std::string framePath(const std::string& pattern, int i) {
    char buf[1024];
    std::snprintf(buf, sizeof buf, pattern.c_str(), i);
    return buf;
}

// ---------------------------------------------------------------- timing -----
struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(a, s)); }
    float stopMs(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

// ------------------------------------------------------------------ main -----
int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s \"frames/frame_%%04d.obj\" numFrames out.csv "
                             "[--cam-res N] [--rnd-rays N] [--repeats R] [--first F] [--label NAME]\n", argv[0]);
        return 1;
    }
    std::string pattern = argv[1], csvPath = argv[3];
    int numFrames = std::atoi(argv[2]);
    int camRes = 1024, rndCount = 1 << 20, repeats = 5, first = 0;
    std::string label;
    for (int i = 4; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cam-res") camRes = std::atoi(argv[++i]);
        else if (a == "--rnd-rays") rndCount = std::atoi(argv[++i]);
        else if (a == "--repeats") repeats = std::atoi(argv[++i]);
        else if (a == "--first") first = std::atoi(argv[++i]);
        else if (a == "--label") label = argv[++i];
    }

    // ---- load all frames on the host (not timed) ----
    std::vector<int3> tris;
    std::vector<std::vector<float3>> frames(numFrames);
    for (int f = 0; f < numFrames; ++f) {
        std::string p = framePath(pattern, first + f);
        if (!loadFrame(p, frames[f], f == 0 ? &tris : nullptr)) return 1;
        if (frames[f].size() != frames[0].size()) {
            std::fprintf(stderr, "%s: vertex count %zu != frame 0 (%zu); topology must be constant\n",
                         p.c_str(), frames[f].size(), frames[0].size());
            return 1;
        }
        std::fprintf(stderr, "\rloaded %d/%d frames", f + 1, numFrames);
    }
    const int nV = (int)frames[0].size(), nT = (int)tris.size();
    std::fprintf(stderr, "\n%d verts, %d tris per frame\n", nV, nT);

    // ---- fixed rays, placed relative to frame 0's bounds ----
    float3 lo = frames[0][0], hi = frames[0][0];
    for (const auto& v : frames[0]) {
        lo = make_float3(fminf(lo.x, v.x), fminf(lo.y, v.y), fminf(lo.z, v.z));
        hi = make_float3(fmaxf(hi.x, v.x), fmaxf(hi.y, v.y), fmaxf(hi.z, v.z));
    }
    float3 center = (lo + hi) * 0.5f;
    float extent = fmaxf(hi.x - lo.x, fmaxf(hi.y - lo.y, hi.z - lo.z));

    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

    float3* dVerts; int3* dTris; Ray *dCam, *dRnd; Hit *dHitsA, *dHitsB; unsigned int* dCounters;
    const int nCam = camRes * camRes;
    const int nMaxRays = std::max(nCam, rndCount);
    CUDA_CHECK(cudaMalloc(&dVerts, sizeof(float3) * nV));
    CUDA_CHECK(cudaMalloc(&dTris, sizeof(int3) * nT));
    CUDA_CHECK(cudaMalloc(&dCam, sizeof(Ray) * nCam));
    CUDA_CHECK(cudaMalloc(&dRnd, sizeof(Ray) * rndCount));
    CUDA_CHECK(cudaMalloc(&dHitsA, sizeof(Hit) * nMaxRays));
    CUDA_CHECK(cudaMalloc(&dHitsB, sizeof(Hit) * nMaxRays));
    CUDA_CHECK(cudaMalloc(&dCounters, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemcpy(dTris, tris.data(), sizeof(int3) * nT, cudaMemcpyHostToDevice));

    {
        float3 eye = center + make_float3(0.f, 0.1f * extent, 2.0f * extent);
        float3 fwd = norm3(center - eye);
        float3 right = norm3(cross3(fwd, make_float3(0, 1, 0)));
        float3 up = cross3(right, fwd);
        dim3 blk(16, 16), grd((camRes + 15) / 16, (camRes + 15) / 16);
        genCameraRays<<<grd, blk, 0, s>>>(dCam, camRes, camRes, eye, fwd, right, up, tanf(22.5f * 3.14159265f / 180.f));
        genRandomRays<<<(rndCount + 255) / 256, 256, 0, s>>>(dRnd, rndCount, center, 0.5f * extent, 1234u);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    // Strategy "rebuild": build from scratch every frame (quality/cost reference).
    // Strategy "update":  build at frame 0, then update() every frame.
    UserBVH bvhRebuild, bvhUpdate;
    bvhRebuild.init(nV, nT);
    bvhUpdate.init(nV, nT);

    FILE* csv = std::fopen(csvPath.c_str(), "w");
    std::fprintf(csv, "label,frame,strategy,update_ms,sah,cam_ms,cam_mrays_s,cam_visits,"
                      "rnd_ms,rnd_mrays_s,rnd_visits,frame_ms,rnd_hits,mismatch_vs_rebuild\n");

    GpuTimer timer;
    auto traceTimed = [&](UserBVH& bvh, const Ray* rays, Hit* hits, int n) {
        bvh.trace(rays, hits, n, dVerts, dTris, s);              // warm-up
        CUDA_CHECK(cudaGetLastError());
        timer.start(s);
        for (int r = 0; r < repeats; ++r) bvh.trace(rays, hits, n, dVerts, dTris, s);
        return timer.stopMs(s) / repeats;
    };

    // Outer NVTX range: names the whole run (e.g. the animation mode) on the
    // timeline, so a report identifies itself whatever its file is called.
    std::string runName = (label.empty() ? std::string("run") : label) + " | " + std::to_string(nV) +
                          " verts, " + std::to_string(nT) + " tris, " + std::to_string(numFrames) + " frames";
    {
    NvtxRange runRange(runName.c_str());
    for (int f = 0; f < numFrames; ++f) {
        char frameName[160];
        std::snprintf(frameName, sizeof frameName, "%s%sframe %d", label.c_str(), label.empty() ? "" : " ", f);
        NvtxRange frameRange(frameName);
        // Upload this frame's positions (not timed: simulates the animation
        // system having already produced them on the GPU).
        CUDA_CHECK(cudaMemcpyAsync(dVerts, frames[f].data(), sizeof(float3) * nV, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        struct Row { float upd, cam, rnd; double sah, camVis, rndVis; };
        Row rows[2];
        UserBVH* bvhs[2] = {&bvhRebuild, &bvhUpdate};
        Hit* rndHits[2] = {dHitsA, dHitsB};

        for (int k = 0; k < 2; ++k) {
            UserBVH& bvh = *bvhs[k];
            timer.start(s);
            NvtxRange strategyRange(k == 0 ? "rebuild" : "update");
            if (k == 0 || f == 0) { NvtxRange r("build");  bvh.build(dVerts, dTris, nV, nT, s); }
            else                  { NvtxRange r("refit");  bvh.update(dVerts, dTris, nV, nT, s); }
            rows[k].upd = timer.stopMs(s);
            CUDA_CHECK(cudaGetLastError());

            { NvtxRange r("trace camera rays"); rows[k].cam = traceTimed(bvh, dCam, rndHits[k], nCam); }  // hits overwritten below
            { NvtxRange r("trace random rays"); rows[k].rnd = traceTimed(bvh, dRnd, rndHits[k], rndCount); }
            NvtxRange diagRange("diagnostics (untimed)");
            rows[k].sah = bvh.sahCost(s);
            rows[k].camVis = bvh.visitsPerRay(dCam, nCam, dVerts, dTris, s);
            rows[k].rndVis = bvh.visitsPerRay(dRnd, rndCount, dVerts, dTris, s);
        }

        // Correctness: random-ray hits of update vs rebuild.
        unsigned int counters[2] = {0, 0};
        CUDA_CHECK(cudaMemsetAsync(dCounters, 0, sizeof(unsigned int) * 2, s));
        compareHits<<<(rndCount + 255) / 256, 256, 0, s>>>(dHitsA, dHitsB, rndCount, 1e-4f, dCounters, dCounters + 1);
        CUDA_CHECK(cudaMemcpyAsync(counters, dCounters, sizeof counters, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        const char* names[2] = {"rebuild", "update"};
        for (int k = 0; k < 2; ++k) {
            const Row& r = rows[k];
            std::fprintf(csv, "%s,%d,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.2f,%.2f,%.4f,%u,%u\n",
                         label.c_str(), f, names[k], r.upd, r.sah,
                         r.cam, nCam / (r.cam * 1e3), r.camVis,
                         r.rnd, rndCount / (r.rnd * 1e3), r.rndVis,
                         r.upd + r.cam + r.rnd,
                         k == 0 ? counters[1] : 0u, k == 1 ? counters[0] : 0u);
        }
        std::fprintf(stderr, "frame %3d | rebuild %8.3f ms, rnd %7.1f Mrays/s | update %8.3f ms, rnd %7.1f Mrays/s | mismatches %u\n",
                     f, rows[0].upd, rndCount / (rows[0].rnd * 1e3),
                     rows[1].upd, rndCount / (rows[1].rnd * 1e3), counters[0]);
    }
    }  // runRange
    std::fclose(csv);

    bvhRebuild.destroy(); bvhUpdate.destroy();
    cudaFree(dVerts); cudaFree(dTris); cudaFree(dCam); cudaFree(dRnd);
    cudaFree(dHitsA); cudaFree(dHitsB); cudaFree(dCounters);
    cudaStreamDestroy(s);
    std::fprintf(stderr, "wrote %s\n", csvPath.c_str());
    return 0;
}
