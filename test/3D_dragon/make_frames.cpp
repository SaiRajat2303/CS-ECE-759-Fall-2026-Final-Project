// make_frames.cpp — turn a static mesh (e.g. Stanford dragon_vrip.ply) into an
// animated frame sequence with constant topology, for bvh_cuda_bench.
//
// Usage: make_frames <in.ply|in.obj> <rigid|twist|wave|explode> <numFrames> <outDir>
//                    [--amount k] [--format ply|obj]
// Writes <outDir>/frame_0000.ply, frame_0001.ply, ...  (binary PLY by default)
#define MESH_IO_IMPLEMENTATION
#include "mesh_io.h"
#include "dynamic_scene.h"   // only the Animator is used; loading goes through mesh_io
#include <cstdio>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " in.(ply|obj) rigid|twist|wave|explode numFrames outDir"
                  << " [--amount k] [--format ply|obj]\n";
        return 1;
    }
    try {
        std::string in = argv[1], outDir = argv[4], fmt = "ply";
        dyn::Mode mode = dyn::parseMode(argv[2]);
        int n = std::stoi(argv[3]);
        float amount = 1.0f;
        for (int i = 5; i + 1 < argc; ++i) {
            std::string a = argv[i];
            if (a == "--amount") amount = std::stof(argv[++i]);
            else if (a == "--format") fmt = argv[++i];
        }
        std::filesystem::create_directories(outDir);

        meshio::TriMesh src = meshio::load(in);
        std::cerr << "loaded " << src.positions.size() << " verts, " << src.triangles.size() << " tris\n";

        dyn::Mesh m;
        m.verts.reserve(src.positions.size());
        for (const auto& p : src.positions) m.verts.emplace_back(p[0], p[1], p[2]);
        for (const auto& t : src.triangles) m.tris.push_back({(uint32_t)t[0], (uint32_t)t[1], (uint32_t)t[2]});
        m.normalizeToUnitBox();

        dyn::Animator anim(m, mode, amount);
        meshio::TriMesh out;
        for (const auto& t : anim.tris()) out.triangles.push_back({(int32_t)t[0], (int32_t)t[1], (int32_t)t[2]});
        out.positions.resize(anim.restVerts().size());

        for (int f = 0; f < n; ++f) {
            auto pos = anim.frame(f, n);
            for (size_t i = 0; i < pos.size(); ++i) out.positions[i] = {pos[i].x, pos[i].y, pos[i].z};
            char name[64];
            std::snprintf(name, sizeof name, "frame_%04d.%s", f, fmt.c_str());
            meshio::save((std::filesystem::path(outDir) / name).string(), out);
            std::cerr << "\rwrote " << f + 1 << "/" << n;
        }
        std::cerr << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}