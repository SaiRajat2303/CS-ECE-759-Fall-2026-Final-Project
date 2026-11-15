#pragma once
#include "cuda_check.h"
#include <chrono>
#include <vector>

/* Marks go into one stream back to back with no sync between them, so timing never serializes stages. One sync at the end, then read the deltas. */
class EventChain {
public:
    explicit EventChain(int capacity) : ev_(capacity) {
        for (auto& e : ev_) CUDA_CHECK(cudaEventCreate(&e));
    }
    ~EventChain() {
        for (auto& e : ev_) cudaEventDestroy(e);
    }
    EventChain(const EventChain&) = delete;
    EventChain& operator=(const EventChain&) = delete;

    void reset() { used_ = 0; }

    void mark(cudaStream_t s) {
        if (used_ >= (int)ev_.size()) {
            std::fprintf(stderr, "EventChain overflow: capacity %zu\n", ev_.size());
            std::exit(1);
        }
        CUDA_CHECK(cudaEventRecord(ev_[used_++], s));
    }

    void sync() {
        if (used_ > 0) CUDA_CHECK(cudaEventSynchronize(ev_[used_ - 1]));
    }

    float span_ms(int a, int b) const {
        if (a < 0 || b >= used_ || a > b) {
            std::fprintf(stderr, "EventChain bad span %d..%d of %d marks\n", a, b, used_);
            std::exit(1);
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_[a], ev_[b]));
        return ms;
    }

    int marks() const { return used_; }

private:
    std::vector<cudaEvent_t> ev_;
    int used_ = 0;
};

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point beg;

    void start() { beg = std::chrono::high_resolution_clock::now(); }

    double stop_ms() const {
        auto d = std::chrono::high_resolution_clock::now() - beg;
        return std::chrono::duration<double, std::milli>(d).count();
    }
};
