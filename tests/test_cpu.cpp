// test_cpu.cpp -- runs the kernel bodies on the CPU and checks the result
// against the independent reference builder. No GPU needed.
//
// Build twice, with -DLBVH_LAYOUT_SOA=0 and =1 (see `make test`).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "lbvh/export.h"
#include "lbvh/kernels_impl.h"
#include "lbvh/pointgen.h"
#include "lbvh/tree.h"

using namespace lbvh;

namespace {

int failures = 0;

void fail(const char* what, Dist dist, int n, uint64_t seed, const std::string& why) {
  std::printf("FAIL %s: dist=%s n=%d seed=%llu: %s\n", what, distName(dist), n,
              static_cast<unsigned long long>(seed), why.c_str());
  ++failures;
}

std::vector<int> shuffledRange(int count, Rng& rng) {
  std::vector<int> order(count);
  std::iota(order.begin(), order.end(), 0);
  for (int i = count - 1; i > 0; --i) std::swap(order[i], order[rng.next() % (i + 1)]);
  return order;
}

// Runs the five build stages on the CPU by calling the kernel bodies once per
// "thread", in random order (the result must not depend on the order).
// With numThreads > 1 the refit stage runs on real threads, which exercises
// the atomic hand-off between sibling subtrees.
void emulateBuild(const std::vector<Vec3>& pts, uint64_t seed, int numThreads, HostTree& out,
                  std::vector<char>& nodeStorage) {
  const int n = static_cast<int>(pts.size());
  Rng rng(seed ^ 0xABCDEFull);
  out.resize(n);

  // Stage 1: scene bounds (CUB reduction on the GPU).
  AABB scene = emptyBox();
  for (int i : shuffledRange(n, rng)) scene = merge(scene, pointBox(pts[i]));
  out.scene = scene;

  // Stage 2: Morton codes.
  std::vector<uint32_t> keys(n), vals(n);
  for (int i : shuffledRange(n, rng)) mortonBody(i, pts.data(), &scene, keys.data(), vals.data());

  // Stage 3: stable sort of (key, value) pairs (CUB radix sort on the GPU).
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
  for (int i = 0; i < n; ++i) {
    out.keys[i] = keys[order[i]];
    out.prims[i] = vals[order[i]];
  }

  // Node storage starts as garbage so that any field the kernels forget to
  // write shows up as a mismatch.
  const int numNodes = 2 * n - 1;
  nodeStorage.assign(nodeStorageBytes(numNodes), static_cast<char>(0xCD));
  const NodeView view = makeNodeView(nodeStorage.data(), numNodes);
  std::vector<int> flags(n - 1, 0x5A5A5A5A);

  // Stage 4: hierarchy.
  for (int i : shuffledRange(n - 1, rng)) hierarchyBody(i, out.keys.data(), n, view, flags.data());

  // Stage 5: refit.
  const std::vector<int> leafOrder = shuffledRange(n, rng);
  if (numThreads <= 1) {
    for (int i : leafOrder) refitBody(i, n, pts.data(), out.prims.data(), view, flags.data());
  } else {
    std::vector<std::thread> workers;
    for (int w = 0; w < numThreads; ++w) {
      workers.emplace_back([&, w] {
        for (size_t k = w; k < leafOrder.size(); k += numThreads)
          refitBody(leafOrder[k], n, pts.data(), out.prims.data(), view, flags.data());
      });
    }
    for (auto& t : workers) t.join();
  }

  for (int f : flags) {
    if (f != 2) {
      std::printf("FAIL: an internal node was visited %d times instead of 2\n", f);
      ++failures;
      break;
    }
  }
  exportNodes(view, n, out);
}

void checkBuild(Dist dist, int n, uint64_t seed, int numThreads) {
  const std::vector<Vec3> pts = generatePoints(dist, n, seed);
  HostTree got, want;
  std::vector<char> storage;
  emulateBuild(pts, seed, numThreads, got, storage);
  buildReference(pts.data(), n, want);

  std::string why;
  if (!validateTree(got, why)) fail("validate", dist, n, seed, why);
  if (!compareTrees(got, want, why)) fail("compare", dist, n, seed, why);

  // Query check against brute force on a few query points.
  const NodeView view = makeNodeView(storage.data(), 2 * n - 1);
  Rng rng(seed + 99);
  for (int q = 0; q < 8; ++q) {
    const Vec3 query = pts[rng.next() % n];
    const float radius = 0.02f + 0.1f * rng.uniform();
    const float r2 = radius * radius;
    int brute = 0;
    for (const Vec3& p : pts) {
      if (distanceSquared(query, pointBox(p)) <= r2) ++brute;
    }
    const int fromTree = radiusCountBody(view, query, r2);
    if (brute != fromTree) {
      fail("query", dist, n, seed, "tree count " + std::to_string(fromTree) + " != brute force " +
                                       std::to_string(brute));
      break;
    }
  }
}

}  // namespace

int main() {
  const Dist dists[] = {Dist::Uniform, Dist::Gaussian, Dist::Clusters, Dist::Surface,
                        Dist::Duplicates};
  const int sizes[] = {2, 3, 4, 5, 7, 8, 16, 17, 31, 100, 257, 1000, 4096, 20000};
  int cases = 0;
  for (Dist d : dists) {
    for (int n : sizes) {
      for (uint64_t seed = 1; seed <= 4; ++seed) {
        checkBuild(d, n, seed, 1);
        ++cases;
      }
    }
    checkBuild(d, 200000, 7, 1);
    checkBuild(d, 200000, 8, 4);  // refit on 4 real threads
    checkBuild(d, 50000, 9, 8);
    cases += 3;
  }

  // Degenerate inputs: all points identical; points on a line; points on a plane.
  {
    HostTree got, want;
    std::vector<char> storage;
    std::string why;
    std::vector<Vec3> same(1000, Vec3{0.25f, 0.5f, 0.75f});
    std::vector<Vec3> line(1000), plane(1000);
    Rng rng(5);
    for (auto& p : line) p = {rng.uniform(), 0.5f, 0.5f};
    for (auto& p : plane) p = {rng.uniform(), rng.uniform(), -3.0f};
    const std::vector<Vec3>* inputs[] = {&same, &line, &plane};
    const char* names[] = {"identical", "line", "plane"};
    for (int k = 0; k < 3; ++k) {
      emulateBuild(*inputs[k], 11, 1, got, storage);
      buildReference(inputs[k]->data(), 1000, want);
      if (!validateTree(got, why) || !compareTrees(got, want, why)) {
        std::printf("FAIL degenerate input '%s': %s\n", names[k], why.c_str());
        ++failures;
      }
      ++cases;
    }
  }

  std::printf("layout=%s: %d cases, %d failures\n", layoutName(), cases, failures);
  return failures == 0 ? 0 : 1;
}
