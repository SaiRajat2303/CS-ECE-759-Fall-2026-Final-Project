//   .obj -> tinyobjloader
//   .ply -> happly (ASCII and binary; handles the Stanford 3D Scanning
//           Repository files, which carry extra vertex properties such as
//           "confidence" and "intensity" that are simply ignored)
//
// Host-only. Exactly ONE .cpp/.cu file must #define MESH_IO_IMPLEMENTATION
// before including this header (it pulls in the tinyobjloader implementation).
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace meshio {

struct TriMesh {
    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<int32_t, 3>> triangles;
};

// Throws std::runtime_error on failure. Polygons are fan-triangulated.
TriMesh load(const std::string& path);

// Binary PLY (fast to reload) or OBJ, chosen by extension.
void save(const std::string& path, const TriMesh& mesh);

} // namespace meshio

#ifdef MESH_IO_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "happly.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace meshio {

static std::string lowerExt(const std::string& path) {
    auto dot = path.find_last_of('.');
    std::string e = dot == std::string::npos ? "" : path.substr(dot + 1);
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return e;
}

static void appendFan(std::vector<std::array<int32_t, 3>>& tris, const std::vector<int32_t>& poly) {
    for (size_t k = 1; k + 1 < poly.size(); ++k) tris.push_back({poly[0], poly[k], poly[k + 1]});
}

static TriMesh loadObj(const std::string& path) {
    tinyobj::ObjReaderConfig cfg;
    cfg.triangulate = true;
    cfg.vertex_color = false;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, cfg))
        throw std::runtime_error("tinyobjloader: " + path + ": " + reader.Error());
    const auto& a = reader.GetAttrib();
    TriMesh m;
    m.positions.resize(a.vertices.size() / 3);
    for (size_t i = 0; i < m.positions.size(); ++i)
        m.positions[i] = {a.vertices[3 * i], a.vertices[3 * i + 1], a.vertices[3 * i + 2]};
    for (const auto& sh : reader.GetShapes()) {
        const auto& idx = sh.mesh.indices;
        for (size_t i = 0; i + 2 < idx.size(); i += 3)
            m.triangles.push_back({idx[i].vertex_index, idx[i + 1].vertex_index, idx[i + 2].vertex_index});
    }
    return m;
}

static TriMesh loadPly(const std::string& path) {
    happly::PLYData ply(path);
    TriMesh m;
    auto pos = ply.getVertexPositions();          // vector<array<double,3>>
    m.positions.resize(pos.size());
    for (size_t i = 0; i < pos.size(); ++i)
        m.positions[i] = {(float)pos[i][0], (float)pos[i][1], (float)pos[i][2]};
    auto faces = ply.getFaceIndices<int32_t>();   // handles vertex_indices / vertex_index
    m.triangles.reserve(faces.size());
    for (const auto& f : faces) appendFan(m.triangles, f);
    return m;
}

TriMesh load(const std::string& path) {
    std::string e = lowerExt(path);
    TriMesh m;
    if (e == "obj") m = loadObj(path);
    else if (e == "ply") m = loadPly(path);
    else throw std::runtime_error("unsupported mesh format: " + path);
    if (m.positions.empty() || m.triangles.empty())
        throw std::runtime_error("mesh has no triangles: " + path);
    return m;
}

void save(const std::string& path, const TriMesh& m) {
    std::string e = lowerExt(path);
    if (e == "ply") {
        happly::PLYData ply;
        const size_t n = m.positions.size();
        std::vector<float> xs(n), ys(n), zs(n);
        for (size_t i = 0; i < n; ++i) { xs[i] = m.positions[i][0]; ys[i] = m.positions[i][1]; zs[i] = m.positions[i][2]; }
        ply.addElement("vertex", n);
        auto& ve = ply.getElement("vertex");
        ve.addProperty<float>("x", xs);
        ve.addProperty<float>("y", ys);
        ve.addProperty<float>("z", zs);
        std::vector<std::vector<int32_t>> faces(m.triangles.size());
        for (size_t i = 0; i < faces.size(); ++i) faces[i] = {m.triangles[i][0], m.triangles[i][1], m.triangles[i][2]};
        ply.addFaceIndices(faces);
        ply.write(path, happly::DataFormat::Binary);
    } else if (e == "obj") {
        std::ofstream out(path);
        for (const auto& p : m.positions) out << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        for (const auto& t : m.triangles) out << "f " << t[0] + 1 << ' ' << t[1] + 1 << ' ' << t[2] + 1 << '\n';
    } else {
        throw std::runtime_error("unsupported output format: " + path);
    }
}

} // namespace meshio
#endif