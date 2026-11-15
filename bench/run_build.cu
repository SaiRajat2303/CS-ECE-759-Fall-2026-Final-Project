#include "lbvh/common.h"
#include "lbvh/export.h"
#include "lbvh/nodes.h"
#include "lbvh/pointgen.h"
#include "lbvh/tree.h"

#include "bounds.cuh"
#include "config.h"
#include "cuda_check.h"
#include "kernels.cuh"
#include "ply.h"
#include "prims.h"
#include "provenance.h"
#include "record.h"
#include "sort.cuh"
#include "timer.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct SceneSpec {
    bool mesh = false;
    lbvh::Dist dist = lbvh::Dist::Uniform;
    int n_log2 = 0;
    std::string path;
};

struct Args {
    std::vector<SceneSpec> scenes;
    int block = 256;
    int repeats = 10;
    double warmup_ms = 300.0;
    uint64_t seed = 1;
    std::string out;
    bool validate = false;
    bool chain = true;
    bool ends = true;
    int device = 0;
};

void usage() {
    std::printf(
        "lbvh_build_%s  full LBVH build with per-stage timing\n"
        "  --points DIST:K[-K2]   random points, n = 2^K (repeatable); DIST is\n"
        "                         uniform | gaussian | clusters | surface | duplicates\n"
        "  --mesh FILE.ply        triangle mesh, one box per triangle (repeatable)\n"
        "  --block 256            threads per block for our kernels\n"
        "  --repeats 10           timed repeats per scene and timing mode\n"
        "  --warmup-ms 300        warm up by wall time, not iteration count\n"
        "  --seed 1               point seed\n"
        "  --out PATH             csv path (default data/full_<host>_<job>_<task>.csv)\n"
        "  --timing-mode both     chain | ends | both\n"
        "  --validate             compare every tree against the CPU reference\n"
        "  --device 0             cuda device index\n",
        lbvh::layoutName());
}

bool parse_points(const std::string& s, std::vector<SceneSpec>& out) {
    size_t c = s.find(':');
    if (c == std::string::npos) return false;
    lbvh::Dist d;
    if (!lbvh::parseDist(s.substr(0, c), d)) return false;
    std::string range = s.substr(c + 1);
    size_t dash = range.find('-');
    int lo = 0, hi = 0;
    try {
        lo = std::stoi(range.substr(0, dash));
        hi = (dash == std::string::npos) ? lo : std::stoi(range.substr(dash + 1));
    } catch (...) {
        return false;
    }
    if (lo < 1 || hi > 27 || lo > hi) return false;
    for (int k = lo; k <= hi; ++k) {
        SceneSpec sc;
        sc.dist = d;
        sc.n_log2 = k;
        out.push_back(sc);
    }
    return true;
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--points") {
            std::string v = next("--points");
            if (!parse_points(v, a.scenes)) { std::fprintf(stderr, "bad --points %s\n", v.c_str()); return false; }
        }
        else if (k == "--mesh") {
            SceneSpec sc;
            sc.mesh = true;
            sc.path = next("--mesh");
            a.scenes.push_back(sc);
        }
        else if (k == "--block")      a.block = std::stoi(next("--block"));
        else if (k == "--repeats")    a.repeats = std::stoi(next("--repeats"));
        else if (k == "--warmup-ms")  a.warmup_ms = std::stod(next("--warmup-ms"));
        else if (k == "--seed")       a.seed = std::stoull(next("--seed"));
        else if (k == "--out")        a.out = next("--out");
        else if (k == "--validate")   a.validate = true;
        else if (k == "--device")     a.device = std::stoi(next("--device"));
        else if (k == "--timing-mode") {
            std::string m = next("--timing-mode");
            a.chain = (m == "chain" || m == "both");
            a.ends  = (m == "ends"  || m == "both");
            if (!a.chain && !a.ends) { std::fprintf(stderr, "bad --timing-mode\n"); return false; }
        }
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else { std::fprintf(stderr, "unknown flag %s\n", k.c_str()); usage(); return false; }
    }
    if (a.scenes.empty()) { std::fprintf(stderr, "no scenes: pass --points or --mesh\n"); return false; }
    if (a.block < 32 || a.block > 1024 || (a.block % 32) != 0) {
        std::fprintf(stderr, "--block must be a multiple of 32 in [32,1024]\n");
        return false;
    }
    if (a.repeats < 1) { std::fprintf(stderr, "--repeats must be >= 1\n"); return false; }
    return true;
}

/* One scene in host memory: either points or triangle boxes, never both. */
struct HostScene {
    std::string name;
    std::vector<lbvh::Vec3> points;
    std::vector<lbvh::AABB> boxes;
    bool is_mesh() const { return !boxes.empty(); }
    int n() const { return (int)(is_mesh() ? boxes.size() : points.size()); }
    const void* data() const { return is_mesh() ? (const void*)boxes.data() : (const void*)points.data(); }
    size_t bytes() const { return is_mesh() ? boxes.size() * sizeof(lbvh::AABB) : points.size() * sizeof(lbvh::Vec3); }
};

struct DeviceBuffers {
    lbvh::Vec3*    points    = nullptr;
    lbvh::AABB*    boxes     = nullptr;
    lbvh::AABB*    scene     = nullptr;
    uint32_t*      codes_in  = nullptr;
    uint32_t*      codes_out = nullptr;
    uint32_t*      idx_in    = nullptr;
    uint32_t*      idx_out   = nullptr;
    void*          node_mem  = nullptr;
    lbvh::NodeView nodes{};
    int*           flags     = nullptr;
    void*          temp      = nullptr;
    size_t         temp_bytes = 0;

    void alloc(int n, bool mesh) {
        const size_t num_nodes = 2 * (size_t)n - 1;
        size_t reduce_bytes = 0;
        if (mesh) {
            CUDA_CHECK(cudaMalloc(&boxes, (size_t)n * sizeof(lbvh::AABB)));
            reduce_bytes = bounds_temp_bytes(centroid_boxes(boxes), n);
        } else {
            CUDA_CHECK(cudaMalloc(&points, (size_t)n * sizeof(lbvh::Vec3)));
            reduce_bytes = bounds_temp_bytes(point_boxes(points), n);
        }
        CUDA_CHECK(cudaMalloc(&scene, sizeof(lbvh::AABB)));
        CUDA_CHECK(cudaMalloc(&codes_in,  (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&codes_out, (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&idx_in,    (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&idx_out,   (size_t)n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&node_mem, lbvh::nodeStorageBytes(num_nodes)));
        nodes = lbvh::makeNodeView(node_mem, num_nodes);
        CUDA_CHECK(cudaMalloc(&flags, (size_t)(n - 1) * sizeof(int)));
        temp_bytes = std::max(reduce_bytes, sort_temp_bytes(n));
        CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
    }

    void free_all() {
        cudaFree(points); cudaFree(boxes); cudaFree(scene);
        cudaFree(codes_in); cudaFree(codes_out);
        cudaFree(idx_in); cudaFree(idx_out);
        cudaFree(node_mem); cudaFree(flags);
        cudaFree(temp);
        *this = DeviceBuffers{};
    }
};

enum Mark { kBegin, kAfterH2d, kAfterBounds, kAfterMorton, kAfterSort, kAfterHierarchy, kAfterRefit, kMarks };

/* Hierarchy zeroes the refit counters itself, so repeated runs need no memset. */
void run_pipeline(const DeviceBuffers& d, const HostScene& hs, int block,
                  cudaStream_t s, EventChain& chain, bool per_stage) {
    const int n = hs.n();
    const bool mesh = hs.is_mesh();
    const int grid_leaves = (n + block - 1) / block;
    const int grid_internal = (n - 1 + block - 1) / block;
    void* d_prims = mesh ? (void*)d.boxes : (void*)d.points;

    chain.reset();
    chain.mark(s);
    CUDA_CHECK(cudaMemcpyAsync(d_prims, hs.data(), hs.bytes(), cudaMemcpyHostToDevice, s));
    if (per_stage) chain.mark(s);
    if (mesh) CUDA_CHECK(reduce_bounds(d.temp, d.temp_bytes, centroid_boxes(d.boxes), d.scene, n, s));
    else      CUDA_CHECK(reduce_bounds(d.temp, d.temp_bytes, point_boxes(d.points), d.scene, n, s));
    if (per_stage) chain.mark(s);
    if (mesh) morton_box_kernel<<<grid_leaves, block, 0, s>>>(d.boxes, d.scene, n, d.codes_in, d.idx_in);
    else      morton_kernel<<<grid_leaves, block, 0, s>>>(d.points, d.scene, n, d.codes_in, d.idx_in);
    CUDA_CHECK_KERNEL();
    if (per_stage) chain.mark(s);
    CUDA_CHECK(sort_codes(d.temp, d.temp_bytes, d.codes_in, d.codes_out, d.idx_in, d.idx_out, n, s));
    if (per_stage) chain.mark(s);
    hierarchy_kernel<<<grid_internal, block, 0, s>>>(d.codes_out, n, d.nodes, d.flags);
    CUDA_CHECK_KERNEL();
    if (per_stage) chain.mark(s);
    if (mesh) refit_box_kernel<<<grid_leaves, block, 0, s>>>(n, d.boxes, d.idx_out, d.nodes, d.flags);
    else      refit_kernel<<<grid_leaves, block, 0, s>>>(n, d.points, d.idx_out, d.nodes, d.flags);
    CUDA_CHECK_KERNEL();
    chain.mark(s);
    chain.sync();
}

uint64_t now_ms() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

float median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5f * (v[m - 1] + v[m]);
}

/* Replace the reference's point-box leaves with triangle boxes and re-merge bottom up, so boxes can be compared too. */
void rebox(lbvh::HostTree& t, const std::vector<lbvh::AABB>& prim_boxes) {
    const int leaf_base = t.n - 1;
    for (int k = 0; k < t.n; ++k) t.box[leaf_base + k] = prim_boxes[t.prim[leaf_base + k]];
    std::vector<int> preorder, stack{0};
    while (!stack.empty()) {
        const int id = stack.back();
        stack.pop_back();
        if (id >= leaf_base) continue;
        preorder.push_back(id);
        stack.push_back(t.left[id]);
        stack.push_back(t.right[id]);
    }
    for (auto it = preorder.rbegin(); it != preorder.rend(); ++it)
        t.box[*it] = lbvh::merge(t.box[t.left[*it]], t.box[t.right[*it]]);
}

/* Structural check first, then a bit-exact comparison of every key, link and box against the independent CPU build. */
bool validate(const DeviceBuffers& d, const HostScene& hs, std::string& why) {
    const int n = hs.n();
    const size_t num_nodes = 2 * (size_t)n - 1;
    std::vector<unsigned char> node_mem(lbvh::nodeStorageBytes(num_nodes));
    lbvh::HostTree got;
    got.resize(n);
    CUDA_CHECK(cudaMemcpy(node_mem.data(), d.node_mem, node_mem.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got.keys.data(), d.codes_out, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got.prims.data(), d.idx_out, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&got.scene, d.scene, sizeof(lbvh::AABB), cudaMemcpyDeviceToHost));
    lbvh::exportNodes(lbvh::makeNodeView(node_mem.data(), num_nodes), n, got);

    if (!lbvh::validateTree(got, why)) { why = "validateTree: " + why; return false; }
    lbvh::HostTree want;
    if (hs.is_mesh()) {
        std::vector<lbvh::Vec3> centroids(n);
        for (int i = 0; i < n; ++i) centroids[i] = box_centroid(hs.boxes[i]);
        lbvh::buildReference(centroids.data(), n, want);
        rebox(want, hs.boxes);
    } else {
        lbvh::buildReference(hs.points.data(), n, want);
    }
    if (!lbvh::compareTrees(got, want, why)) { why = "compareTrees: " + why; return false; }
    return true;
}

bool load_scene(const SceneSpec& sc, uint64_t seed, HostScene& hs) {
    if (!sc.mesh) {
        hs.name = std::string(lbvh::distName(sc.dist)) + "_s" + std::to_string(seed);
        hs.points = lbvh::generatePoints(sc.dist, 1 << sc.n_log2, seed + (uint64_t)sc.n_log2);
        return true;
    }
    hs.name = std::filesystem::path(sc.path).stem().string();
    PlyMesh mesh;
    std::string err;
    CpuTimer t;
    t.start();
    if (!load_ply(sc.path, mesh, err)) {
        std::fprintf(stderr, "%s: %s\n", sc.path.c_str(), err.c_str());
        return false;
    }
    hs.boxes = triangle_boxes(mesh);
    std::printf("%s: %zu vertices, %zu triangles, loaded in %.0f ms\n", sc.path.c_str(),
                mesh.vertices(), mesh.triangles(), t.stop_ms());
    if (hs.boxes.size() < 2) {
        std::fprintf(stderr, "%s: need at least 2 triangles\n", sc.path.c_str());
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    CUDA_CHECK(cudaSetDevice(a.device));
    Provenance pv = gather_provenance(a.device);

    if (a.out.empty()) {
        std::string task = env_or("SLURM_ARRAY_TASK_ID", "0");
        a.out = "data/full_" + pv.host + "_" + pv.slurm_job + "_" + task + ".csv";
    }
    std::filesystem::path outp(a.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());

    std::printf("gpu=%s %s l2=%zu MB sms=%d cuda_rt=%s host=%s job=%s layout=%s block=%d\n",
                pv.gpu_name.c_str(), pv.arch.c_str(), pv.l2_bytes >> 20, pv.sm_count,
                pv.cuda_rt.c_str(), pv.host.c_str(), pv.slurm_job.c_str(),
                lbvh::layoutName(), a.block);
    std::printf("out=%s\n", a.out.c_str());

    CsvSink sink(a.out);
    if (sink.resumed()) std::printf("resuming: %zu rows already present\n", sink.resumed());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    EventChain chain(kMarks);

    /* Interleave the two timing modes, and shuffle scenes, so clock sag cannot favour one of them. */
    std::vector<std::pair<bool, int>> work;
    for (int r = 0; r < a.repeats; ++r) {
        if (a.chain) work.emplace_back(true, r);
        if (a.ends)  work.emplace_back(false, r);
    }
    lbvh::Rng order_rng(a.seed ^ 0xA5A5A5A5A5A5A5A5ull);
    for (size_t i = work.size(); i > 1; --i) std::swap(work[i - 1], work[order_rng.next() % i]);
    std::vector<SceneSpec> scenes = a.scenes;
    for (size_t i = scenes.size(); i > 1; --i) std::swap(scenes[i - 1], scenes[order_rng.next() % i]);

    int exit_code = 0;

    for (const SceneSpec& sc : scenes) {
        HostScene hs;
        if (!load_scene(sc, a.seed, hs)) { exit_code = 1; continue; }
        const int n = hs.n();

        auto make_row = [&](bool per_stage, int rep) {
            Row row;
            stamp(row, pv);
            row.n = (uint32_t)n;
            row.scene = hs.name;
            row.mem_mode = to_string(MemMode::Pageable);
            row.layout = lbvh::layoutName();
            row.block = a.block;
            row.timing_mode = per_stage ? "chain" : "ends";
            row.repeat = rep;
            row.prim_kind = hs.is_mesh() ? "triangles" : "points";
            return row;
        };

        bool all_done = std::all_of(work.begin(), work.end(),
                                    [&](const auto& w) { return sink.done(make_row(w.first, w.second)); });
        if (all_done && !a.validate) {
            std::printf("%-14s n=%-9d already recorded, skipped\n", hs.name.c_str(), n);
            continue;
        }

        DeviceBuffers d;
        d.alloc(n, hs.is_mesh());

        /* Warm up against a wall-time budget: the idle SM clock on the Ada card is 210 MHz. */
        CpuTimer warm;
        warm.start();
        int warm_iters = 0;
        do {
            run_pipeline(d, hs, a.block, stream, chain, false);
            ++warm_iters;
        } while (warm.stop_ms() < a.warmup_ms);

        std::vector<float> tot_chain, tot_ends, hier, refit;

        for (auto& [per_stage, rep] : work) {
            Row row = make_row(per_stage, rep);
            if (sink.done(row)) continue;

            run_pipeline(d, hs, a.block, stream, chain, per_stage);
            row.ts_ms = now_ms();
            row.warmup_iters = warm_iters;

            if (per_stage) {
                row.ms_h2d       = chain.span_ms(kBegin, kAfterH2d);
                row.ms_bounds    = chain.span_ms(kAfterH2d, kAfterBounds);
                row.ms_morton    = chain.span_ms(kAfterBounds, kAfterMorton);
                row.ms_sort      = chain.span_ms(kAfterMorton, kAfterSort);
                row.ms_hierarchy = chain.span_ms(kAfterSort, kAfterHierarchy);
                row.ms_refit     = chain.span_ms(kAfterHierarchy, kAfterRefit);
                row.ms_total     = chain.span_ms(kBegin, kAfterRefit);
                tot_chain.push_back(row.ms_total);
                hier.push_back(row.ms_hierarchy);
                refit.push_back(row.ms_refit);
            } else {
                row.ms_total = chain.span_ms(0, 1);
                tot_ends.push_back(row.ms_total);
            }
            sink.append(row);
        }

        std::printf("%-14s n=%-9d warmup_iters=%-4d", hs.name.c_str(), n, warm_iters);
        if (!tot_chain.empty())
            std::printf("  total=%8.3f ms  hierarchy=%7.3f  refit=%7.3f",
                        median(tot_chain), median(hier), median(refit));
        if (!tot_ends.empty()) std::printf("  ends_total=%8.3f ms", median(tot_ends));
        std::printf("\n");

        if (a.validate) {
            run_pipeline(d, hs, a.block, stream, chain, true);
            std::string why;
            bool ok = validate(d, hs, why);
            std::printf("  validate %s n=%d %s: %s%s%s\n", hs.name.c_str(), n, lbvh::layoutName(),
                        ok ? "PASS" : "FAIL", ok ? "" : " - ", why.c_str());
            if (!ok) exit_code = 1;
        }

        d.free_all();
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return exit_code;
}
