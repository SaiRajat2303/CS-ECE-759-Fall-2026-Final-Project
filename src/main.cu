// main.cu -- benchmark driver.
//
// For one input point cloud, runs the build under every requested
// combination of the runtime knobs (host memory mode x launch mode), checks
// each result against the CPU reference, and appends one CSV row per timed
// run. The compile-time knobs of this binary are written into every row.
//
// Example:
//   ./lbvh --n 1000000 --dist clusters --seed 3 --reps 5 --warmup 2 \
//          --hostmem all --launch all --verify --csv results.csv
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "lbvh/builder.h"
#include "lbvh/pointgen.h"
#include "lbvh/tree.h"

using namespace lbvh;

namespace {

struct Options {
  int n = 1000000;
  Dist dist = Dist::Uniform;
  uint64_t seed = 1;
  std::string inputFile;  // overrides n/dist/seed when set
  int reps = 5;
  int warmup = 2;
  double warmupMs = 200.0;  // keep warming up until this much time has passed
  std::vector<HostMem> hostMems{HostMem::Pageable, HostMem::Pinned, HostMem::Unified,
                                HostMem::UnifiedPrefetch};
  std::vector<LaunchMode> launches{LaunchMode::Separate, LaunchMode::Graph};
  bool verify = false;
  bool verifyEvery = false;
  bool stages = false;
  bool shuffle = false;
  bool info = false;
  int numQueries = 0;
  float radiusFraction = 0.01f;  // query radius as a fraction of the largest scene extent
  std::string csvPath;
};

void usage() {
  std::printf(
      "lbvh benchmark driver (block=%d layout=%s stream_hint=%s reg_cap=%d)\n"
      "  --n N                 number of points (default 1000000)\n"
      "  --dist NAME           uniform | gaussian | clusters | surface | duplicates\n"
      "  --seed S              seed for the generator\n"
      "  --input FILE          read points from an ASCII .xyz file instead\n"
      "  --reps R              timed builds per configuration (default 5)\n"
      "  --warmup W            untimed builds per configuration, at least (default 2)\n"
      "  --warmup-ms T         and keep warming up for at least T ms (default 200)\n"
      "  --hostmem LIST        all, or comma list of pageable,pinned,unified,unified_prefetch\n"
      "  --launch LIST         all, or comma list of separate,graph\n"
      "  --stages              per-stage times from CUDA events (separate launches only)\n"
      "  --verify              compare every configuration's tree with the CPU reference\n"
      "  --verify-every        like --verify, but check the tree after every timed build\n"
      "  --query M             also time M radius queries on the finished tree\n"
      "  --radius F            query radius as a fraction of the scene extent (default 0.01)\n"
      "  --shuffle             run the configurations in random order\n"
      "  --csv FILE            append results to FILE (default: print to stdout)\n"
      "  --info                print GPU properties and exit\n",
      LBVH_BLOCK_SIZE, layoutName(), streamHintName(), LBVH_REG_CAP);
}

std::vector<std::string> splitList(const std::string& s) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= s.size()) {
    const size_t comma = s.find(',', start);
    const size_t end = comma == std::string::npos ? s.size() : comma;
    if (end > start) parts.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return parts;
}

bool parseOptions(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--n") o.n = std::atoi(value());
    else if (a == "--dist") {
      if (!parseDist(value(), o.dist)) return false;
    } else if (a == "--seed") o.seed = std::strtoull(value(), nullptr, 10);
    else if (a == "--input") o.inputFile = value();
    else if (a == "--reps") o.reps = std::atoi(value());
    else if (a == "--warmup") o.warmup = std::atoi(value());
    else if (a == "--warmup-ms") o.warmupMs = std::atof(value());
    else if (a == "--hostmem") {
      const std::string v = value();
      if (v != "all") {
        o.hostMems.clear();
        for (const auto& part : splitList(v)) {
          HostMem m;
          if (!parseHostMem(part, m)) return false;
          o.hostMems.push_back(m);
        }
      }
    } else if (a == "--launch") {
      const std::string v = value();
      if (v != "all") {
        o.launches.clear();
        for (const auto& part : splitList(v)) {
          LaunchMode m;
          if (!parseLaunchMode(part, m)) return false;
          o.launches.push_back(m);
        }
      }
    } else if (a == "--stages") o.stages = true;
    else if (a == "--verify") o.verify = true;
    else if (a == "--verify-every") o.verify = o.verifyEvery = true;
    else if (a == "--shuffle") o.shuffle = true;
    else if (a == "--info") o.info = true;
    else if (a == "--query") o.numQueries = std::atoi(value());
    else if (a == "--radius") o.radiusFraction = static_cast<float>(std::atof(value()));
    else if (a == "--csv") o.csvPath = value();
    else return false;
  }
  return o.n >= 2 && o.reps >= 1 && o.warmup >= 0 && o.warmupMs >= 0 && !o.hostMems.empty() && !o.launches.empty();
}

const char* kCsvHeader =
    "host,gpu,block_size,layout,stream_hint,reg_cap,host_mem,launch,input,n,seed,"
    "config_order,rep,total_ms,set_points_ms,transfer_ms,bounds_ms,morton_ms,sort_ms,hierarchy_ms,refit_ms,"
    "graph_setup_ms,graph_nodes,verified,query_ms,query_count,query_hits,"
    "extent_x,extent_y,extent_z,dup_frac,cell_frac,max_cell_frac,device_mb\n";

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parseOptions(argc, argv, opt)) {
    usage();
    return 2;
  }

  if (opt.info) {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
      std::fprintf(stderr, "no usable CUDA device\n");
      return 1;
    }
    std::printf("gpu: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    std::printf("memory: %.1f GB, L2 cache: %.1f MB\n", p.totalGlobalMem / 1073741824.0,
                p.l2CacheSize / 1048576.0);
    std::printf("SMs: %d, max threads per SM: %d, registers per SM: %d\n", p.multiProcessorCount,
                p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor);
    std::printf("registers per thread before registers limit occupancy: %d\n",
                p.regsPerMultiprocessor / p.maxThreadsPerMultiProcessor);
    std::printf("this binary: block=%d layout=%s stream_hint=%s reg_cap=%d\n", LBVH_BLOCK_SIZE,
                layoutName(), streamHintName(), LBVH_REG_CAP);
    return 0;
  }

  // Input points.
  std::vector<Vec3> points;
  std::string inputName;
  if (!opt.inputFile.empty()) {
    if (!loadXYZ(opt.inputFile, points) || points.size() < 2) {
      std::fprintf(stderr, "could not read at least 2 points from %s\n", opt.inputFile.c_str());
      return 1;
    }
    inputName = opt.inputFile;
    opt.seed = 0;
  } else {
    points = generatePoints(opt.dist, opt.n, opt.seed);
    inputName = distName(opt.dist);
  }
  const int n = static_cast<int>(points.size());

  // Provenance.
  char host[256] = "unknown";
  gethostname(host, sizeof host - 1);
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::fprintf(stderr, "no usable CUDA device\n");
    return 1;
  }

  // CPU reference tree and input features (only when asked: it costs a
  // CPU sort of the whole input).
  HostTree reference;
  InputFeatures features;
  bool haveReference = false;
  if (opt.verify) {
    buildReference(points.data(), n, reference);
    features = computeFeatures(reference);
    haveReference = true;
  }

  // Query points: the first M input points.
  std::vector<Vec3> queries;
  if (opt.numQueries > 0)
    queries.assign(points.begin(), points.begin() + std::min(opt.numQueries, n));

  // Output.
  std::FILE* out = stdout;
  if (!opt.csvPath.empty()) {
    const bool exists = access(opt.csvPath.c_str(), F_OK) == 0;
    out = std::fopen(opt.csvPath.c_str(), "a");
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", opt.csvPath.c_str());
      return 1;
    }
    if (!exists) std::fputs(kCsvHeader, out);
  } else {
    std::fputs(kCsvHeader, out);
  }

  // The runtime configurations, optionally in random order so that slow
  // drift (clock, temperature) does not favour any one of them.
  struct Config {
    HostMem hostMem;
    LaunchMode launch;
  };
  std::vector<Config> configs;
  for (HostMem m : opt.hostMems)
    for (LaunchMode l : opt.launches) configs.push_back({m, l});
  if (opt.shuffle) {
    Rng rng(opt.seed * 7919 + static_cast<uint64_t>(getpid()));
    for (size_t i = configs.size() - 1; i > 0; --i)
      std::swap(configs[i], configs[rng.next() % (i + 1)]);
  }

  using Clock = std::chrono::steady_clock;
  auto msSince = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };

  int mismatches = 0;
  for (size_t order = 0; order < configs.size(); ++order) {
    const Config& cfg = configs[order];
    Builder builder(n, cfg.hostMem, cfg.launch);

    // Warm-up: at least `warmup` builds and at least `warmupMs` of work, so
    // the GPU clock has come back up after the idle time between
    // configurations (clocks cannot be locked on a shared cluster).
    const auto warmStart = Clock::now();
    for (int w = 0; w < opt.warmup || msSince(warmStart) < opt.warmupMs; ++w) {
      builder.setPoints(points.data());
      builder.build();
    }

    int verified = -1;  // -1 = not checked, 1 = identical to reference, 0 = mismatch
    HostTree tree;
    auto checkTree = [&](int rep) {
      builder.download(tree);
      std::string why;
      const bool same = compareTrees(tree, reference, why);
      if (verified != 0) verified = same ? 1 : 0;
      if (!same) {
        ++mismatches;
        std::fprintf(stderr, "MISMATCH host_mem=%s launch=%s rep=%d: %s\n",
                     hostMemName(cfg.hostMem), launchModeName(cfg.launch), rep, why.c_str());
      }
    };

    // Timed builds. setPoints() is timed separately: for unified memory it
    // includes bringing the pages back from the GPU, a cost the other memory
    // modes do not have.
    std::vector<BuildTimes> runs(opt.reps);
    std::vector<double> setPointsMs(opt.reps);
    for (int r = 0; r < opt.reps; ++r) {
      const auto setStart = Clock::now();
      builder.setPoints(points.data());
      setPointsMs[r] = msSince(setStart);
      runs[r] = builder.build(opt.stages);
      if (opt.verifyEvery) checkTree(r);
    }

    // Check the tree of the last build.
    if (haveReference && !opt.verifyEvery) checkTree(opt.reps - 1);

    // Fixed query kernel.
    double queryMs = -1;
    long long queryHits = -1;
    if (!queries.empty()) {
      if (!haveReference) builder.download(tree);
      const float extent = std::max({tree.scene.hi.x - tree.scene.lo.x,
                                     tree.scene.hi.y - tree.scene.lo.y,
                                     tree.scene.hi.z - tree.scene.lo.z});
      const float radius = opt.radiusFraction * extent;
      std::vector<int> counts;
      builder.radiusQuery(queries, radius, counts);            // warm-up
      queryMs = builder.radiusQuery(queries, radius, counts);  // timed
      queryHits = 0;
      for (int c : counts) queryHits += c;
      if (opt.verify) {  // brute-force check of a few queries
        const int checks = std::min<int>(16, static_cast<int>(queries.size()));
        for (int q = 0; q < checks; ++q) {
          int brute = 0;
          const float r2 = radius * radius;
          for (const Vec3& p : points) {
            const float dx = p.x - queries[q].x, dy = p.y - queries[q].y, dz = p.z - queries[q].z;
            if (dx * dx + dy * dy + dz * dz <= r2) ++brute;
          }
          if (brute != counts[q]) {
            ++mismatches;
            verified = 0;
            std::fprintf(stderr, "QUERY MISMATCH query %d: gpu %d, brute force %d\n", q, counts[q],
                         brute);
            break;
          }
        }
      }
    }

    for (int r = 0; r < opt.reps; ++r) {
      const BuildTimes& t = runs[r];
      std::fprintf(out,
                   "%s,%s,%d,%s,%s,%d,%s,%s,%s,%d,%llu,%zu,%d,"
                   "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                   "%.6f,%zu,%d,%.6f,%zu,%lld,"
                   "%.6g,%.6g,%.6g,%.6f,%.6f,%.6f,%.2f\n",
                   host, prop.name, LBVH_BLOCK_SIZE, layoutName(), streamHintName(), LBVH_REG_CAP,
                   hostMemName(cfg.hostMem), launchModeName(cfg.launch), inputName.c_str(), n,
                   static_cast<unsigned long long>(opt.seed), order, r,
                   t.totalMs, setPointsMs[r], t.transferMs, t.boundsMs, t.mortonMs, t.sortMs, t.hierarchyMs,
                   t.refitMs,
                   builder.graphSetupMs(), builder.graphNodeCount(), verified, queryMs,
                   queries.size(), queryHits,
                   haveReference ? features.extentX : -1.0,
                   haveReference ? features.extentY : -1.0,
                   haveReference ? features.extentZ : -1.0,
                   haveReference ? features.duplicateFraction : -1.0,
                   haveReference ? features.occupiedCellFraction : -1.0,
                   haveReference ? features.maxCellFraction : -1.0,
                   builder.deviceBytes() / (1024.0 * 1024.0));
    }
    std::fflush(out);
  }

  if (out != stdout) std::fclose(out);
  if (mismatches) {
    std::fprintf(stderr, "%d configuration(s) did not match the reference\n", mismatches);
    return 3;
  }
  return 0;
}
