#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/*
 * Minimal PLY reader: ascii, binary_little_endian and binary_big_endian.
 * Keeps vertex x, y, z and the face vertex index lists, fanning polygons into triangles. Every other property and element is read and discarded.
 */
struct PlyMesh {
    std::vector<float>    xyz;
    std::vector<uint32_t> tris;
    size_t vertices() const { return xyz.size() / 3; }
    size_t triangles() const { return tris.size() / 3; }
};

namespace ply_detail {

enum class Format { Ascii, LittleEndian, BigEndian };
enum class Kind { Signed, Unsigned, Float };

struct Scalar {
    int size = 0;
    Kind kind = Kind::Signed;
};

struct Property {
    std::string name;
    Scalar type;
    bool is_list = false;
    Scalar count_type;
};

struct Element {
    std::string name;
    size_t count = 0;
    std::vector<Property> props;
};

inline bool parse_scalar(const std::string& s, Scalar& t) {
    if (s == "char" || s == "int8")         t = {1, Kind::Signed};
    else if (s == "uchar" || s == "uint8")  t = {1, Kind::Unsigned};
    else if (s == "short" || s == "int16")  t = {2, Kind::Signed};
    else if (s == "ushort" || s == "uint16") t = {2, Kind::Unsigned};
    else if (s == "int" || s == "int32")    t = {4, Kind::Signed};
    else if (s == "uint" || s == "uint32")  t = {4, Kind::Unsigned};
    else if (s == "float" || s == "float32") t = {4, Kind::Float};
    else if (s == "double" || s == "float64") t = {8, Kind::Float};
    else return false;
    return true;
}

inline bool host_little_endian() {
    const uint16_t one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

class Reader {
public:
    Reader(std::istream& in, Format f) : in_(in), fmt_(f), swap_(f != Format::Ascii && ((f == Format::LittleEndian) != host_little_endian())) {}

    bool read(const Scalar& t, double& out) {
        if (fmt_ == Format::Ascii) {
            std::string tok;
            if (!(in_ >> tok)) return false;
            try { out = std::stod(tok); } catch (...) { return false; }
            return true;
        }
        unsigned char b[8];
        if (!in_.read(reinterpret_cast<char*>(b), t.size)) return false;
        if (swap_)
            for (int i = 0; i < t.size / 2; ++i) std::swap(b[i], b[t.size - 1 - i]);
        switch (t.kind) {
            case Kind::Float:
                if (t.size == 4) { float v; std::memcpy(&v, b, 4); out = v; }
                else { double v; std::memcpy(&v, b, 8); out = v; }
                return true;
            case Kind::Signed:
                if (t.size == 1) { int8_t v; std::memcpy(&v, b, 1); out = v; }
                else if (t.size == 2) { int16_t v; std::memcpy(&v, b, 2); out = v; }
                else { int32_t v; std::memcpy(&v, b, 4); out = v; }
                return true;
            case Kind::Unsigned:
                if (t.size == 1) { uint8_t v; std::memcpy(&v, b, 1); out = v; }
                else if (t.size == 2) { uint16_t v; std::memcpy(&v, b, 2); out = v; }
                else { uint32_t v; std::memcpy(&v, b, 4); out = v; }
                return true;
        }
        return false;
    }

private:
    std::istream& in_;
    Format fmt_;
    bool swap_;
};

} // namespace ply_detail

inline bool load_ply(const std::string& path, PlyMesh& mesh, std::string& err) {
    using namespace ply_detail;
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open " + path; return false; }

    std::string line;
    if (!std::getline(in, line) || line.rfind("ply", 0) != 0) { err = "not a PLY file"; return false; }

    Format fmt = Format::Ascii;
    std::vector<Element> elems;
    bool ended = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string word;
        ls >> word;
        if (word == "format") {
            std::string f;
            ls >> f;
            if (f == "ascii") fmt = Format::Ascii;
            else if (f == "binary_little_endian") fmt = Format::LittleEndian;
            else if (f == "binary_big_endian") fmt = Format::BigEndian;
            else { err = "unknown format " + f; return false; }
        } else if (word == "element") {
            Element e;
            ls >> e.name >> e.count;
            elems.push_back(e);
        } else if (word == "property") {
            if (elems.empty()) { err = "property before element"; return false; }
            Property p;
            std::string t;
            ls >> t;
            if (t == "list") {
                std::string ct, it;
                ls >> ct >> it >> p.name;
                p.is_list = true;
                if (!parse_scalar(ct, p.count_type) || !parse_scalar(it, p.type)) { err = "bad list type"; return false; }
            } else {
                ls >> p.name;
                if (!parse_scalar(t, p.type)) { err = "bad property type " + t; return false; }
            }
            elems.back().props.push_back(p);
        } else if (word == "end_header") {
            ended = true;
            break;
        }
    }
    if (!ended) { err = "no end_header"; return false; }

    Reader rd(in, fmt);
    mesh.xyz.clear();
    mesh.tris.clear();
    size_t num_vertices = 0;
    std::vector<double> list;
    for (const Element& e : elems) {
        const bool is_vertex = (e.name == "vertex");
        const bool is_face = (e.name == "face");
        if (is_vertex) { num_vertices = e.count; mesh.xyz.assign(3 * e.count, 0.0f); }
        for (size_t row = 0; row < e.count; ++row) {
            for (const Property& p : e.props) {
                double v = 0.0;
                if (!p.is_list) {
                    if (!rd.read(p.type, v)) { err = "truncated data in " + e.name; return false; }
                    if (is_vertex) {
                        if (p.name == "x") mesh.xyz[3 * row] = (float)v;
                        else if (p.name == "y") mesh.xyz[3 * row + 1] = (float)v;
                        else if (p.name == "z") mesh.xyz[3 * row + 2] = (float)v;
                    }
                    continue;
                }
                double cnt = 0.0;
                if (!rd.read(p.count_type, cnt) || cnt < 0) { err = "bad list count in " + e.name; return false; }
                list.resize((size_t)cnt);
                for (double& x : list)
                    if (!rd.read(p.type, x)) { err = "truncated list in " + e.name; return false; }
                if (is_face && (p.name == "vertex_indices" || p.name == "vertex_index")) {
                    for (size_t k = 1; k + 1 < list.size(); ++k) {
                        const double idx[3] = {list[0], list[k], list[k + 1]};
                        for (double id : idx) {
                            if (id < 0 || id >= (double)num_vertices) { err = "face index out of range"; return false; }
                            mesh.tris.push_back((uint32_t)id);
                        }
                    }
                }
            }
        }
    }
    if (mesh.vertices() == 0 || mesh.triangles() == 0) { err = "no vertices or no faces"; return false; }
    return true;
}
