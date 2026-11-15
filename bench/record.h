#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>

constexpr int kSchemaVersion = 3;

struct Row {
    std::string git_commit;
    std::string gpu_name;
    std::string arch;
    std::string cuda_rt;
    std::string host;
    std::string bus_id;
    std::string slurm_job;
    std::string cotenants;
    std::string throttle;
    uint32_t    n = 0;
    std::string scene;
    std::string mem_mode;
    std::string layout;
    int         block = 0;
    std::string timing_mode;
    int         repeat = 0;
    float       ms_h2d = 0.0f;
    float       ms_bounds = 0.0f;
    float       ms_morton = 0.0f;
    float       ms_sort = 0.0f;
    float       ms_hierarchy = 0.0f;
    float       ms_refit = 0.0f;
    float       ms_total = 0.0f;
    uint64_t    ts_ms = 0;
    std::string prim_kind;
    int         warmup_iters = 0;
};

inline std::string csv_sanitize(const std::string& s) {
    if (s.empty()) return "NA";
    std::string out = s;
    for (char& c : out)
        if (c == ',' || c == '\n' || c == '\r' || c == '"') c = ';';
    return out;
}

inline const char* csv_header() {
    return "schema,git_commit,gpu_name,arch,cuda_rt,host,bus_id,slurm_job,cotenants,throttle,"
           "n,scene,mem_mode,layout,block,timing_mode,repeat,"
           "ms_h2d,ms_bounds,ms_morton,ms_sort,ms_hierarchy,ms_refit,ms_total,ts_ms,"
           "prim_kind,warmup_iters\n";
}

/* Key column indices. New non-key columns go at the end so these never move. */
enum : int {
    kColCommit = 1, kColGpu = 2, kColN = 10, kColScene = 11, kColMem = 12,
    kColLayout = 13, kColBlock = 14, kColTimingMode = 15, kColRepeat = 16,
    kColCount = 27
};

inline std::string row_key(const std::string& commit, const std::string& gpu,
                           const std::string& n, const std::string& scene,
                           const std::string& mem, const std::string& layout,
                           const std::string& block, const std::string& tmode,
                           const std::string& repeat) {
    return commit + '|' + gpu + '|' + n + '|' + scene + '|' + mem + '|' + layout +
           '|' + block + '|' + tmode + '|' + repeat;
}

inline std::string row_key(const Row& r) {
    return row_key(csv_sanitize(r.git_commit), csv_sanitize(r.gpu_name),
                   std::to_string(r.n), csv_sanitize(r.scene), r.mem_mode, r.layout,
                   std::to_string(r.block), r.timing_mode, std::to_string(r.repeat));
}

inline std::string row_to_csv(const Row& r) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(6);
    o << kSchemaVersion << ',' << csv_sanitize(r.git_commit) << ','
      << csv_sanitize(r.gpu_name) << ',' << csv_sanitize(r.arch) << ','
      << csv_sanitize(r.cuda_rt) << ',' << csv_sanitize(r.host) << ','
      << csv_sanitize(r.bus_id) << ',' << csv_sanitize(r.slurm_job) << ','
      << csv_sanitize(r.cotenants) << ',' << csv_sanitize(r.throttle) << ','
      << r.n << ',' << csv_sanitize(r.scene) << ',' << r.mem_mode << ',' << r.layout << ','
      << r.block << ',' << r.timing_mode << ',' << r.repeat << ','
      << r.ms_h2d << ',' << r.ms_bounds << ',' << r.ms_morton << ',' << r.ms_sort << ','
      << r.ms_hierarchy << ',' << r.ms_refit << ',' << r.ms_total << ',' << r.ts_ms << ','
      << csv_sanitize(r.prim_kind) << ',' << r.warmup_iters << '\n';
    return o.str();
}

/*
 * One sink per process: concurrent appends from a job array can interleave partial lines on CephFS,
 * so each task writes its own file and analysis merges them.
 */
class CsvSink {
public:
    explicit CsvSink(const std::string& path) : path_(path) {
        bool had_header = false;
        bool file_existed = false;
        {
            std::ifstream in(path_);
            if (in) {
                file_existed = true;
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty()) continue;
                    if (line.rfind("schema,", 0) == 0) { had_header = true; continue; }
                    std::string k = key_from_line(line);
                    if (!k.empty()) done_.insert(k);
                }
            }
        }
        /* A preempted job can leave a final line with no newline; without this the next row fuses onto it. */
        bool unterminated = false;
        {
            std::ifstream in(path_, std::ios::binary | std::ios::ate);
            if (in) {
                std::streamoff sz = in.tellg();
                if (sz > 0) {
                    in.seekg(sz - 1);
                    char c = 0;
                    in.get(c);
                    unterminated = (c != '\n');
                }
            }
        }

        out_.open(path_, std::ios::app);
        if (!out_) {
            std::fprintf(stderr, "cannot open %s for append\n", path_.c_str());
            std::exit(1);
        }
        if (unterminated) out_ << '\n' << std::flush;
        if (!had_header && (!file_existed || done_.empty())) out_ << csv_header() << std::flush;
    }

    bool done(const Row& r) const { return done_.count(row_key(r)) > 0; }

    void append(const Row& r) {
        out_ << row_to_csv(r) << std::flush;
        done_.insert(row_key(r));
    }

    size_t resumed() const { return done_.size(); }

private:
    static std::string key_from_line(const std::string& line) {
        std::string f[kColCount];
        std::stringstream ss(line);
        int got = 0;
        while (got < kColCount && std::getline(ss, f[got], ',')) ++got;
        if (got < kColCount || f[0] != std::to_string(kSchemaVersion)) return {};
        return row_key(f[kColCommit], f[kColGpu], f[kColN], f[kColScene], f[kColMem],
                       f[kColLayout], f[kColBlock], f[kColTimingMode], f[kColRepeat]);
    }

    std::string path_;
    std::ofstream out_;
    std::unordered_set<std::string> done_;
};
