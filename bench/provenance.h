#pragma once
#include "cuda_check.h"
#include "record.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#ifndef GIT_COMMIT
#define GIT_COMMIT "nogit"
#endif

inline std::string env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : std::string(fallback);
}

struct Provenance {
    std::string git_commit;
    std::string gpu_name;
    std::string arch;
    std::string cuda_rt;
    std::string host;
    std::string bus_id;
    std::string slurm_job;
    std::string cotenants;
    std::string throttle;
    size_t      l2_bytes = 0;
    int         sm_count = 0;
};

/* Clock and co-tenancy come from the job script via env: forking nvidia-smi from this process would perturb the timings. */
inline Provenance gather_provenance(int device) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));

    char bus[32] = {0};
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));

    int rt = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));

    char hn[256] = {0};
    if (gethostname(hn, sizeof(hn) - 1) != 0) std::snprintf(hn, sizeof(hn), "unknown");

    Provenance pv;
    pv.git_commit = GIT_COMMIT;
    pv.gpu_name   = p.name;
    pv.arch       = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
    pv.cuda_rt    = std::to_string(rt);
    pv.host       = hn;
    pv.bus_id     = bus;
    pv.slurm_job  = env_or("SLURM_JOB_ID", "local");
    pv.cotenants  = env_or("LBVH_COTENANTS", "NA");
    pv.throttle   = env_or("LBVH_THROTTLE", "NA");
    pv.l2_bytes   = (size_t)p.l2CacheSize;
    pv.sm_count   = p.multiProcessorCount;
    return pv;
}

inline void stamp(Row& r, const Provenance& pv) {
    r.git_commit = pv.git_commit;
    r.gpu_name   = pv.gpu_name;
    r.arch       = pv.arch;
    r.cuda_rt    = pv.cuda_rt;
    r.host       = pv.host;
    r.bus_id     = pv.bus_id;
    r.slurm_job  = pv.slurm_job;
    r.cotenants  = pv.cotenants;
    r.throttle   = pv.throttle;
}
