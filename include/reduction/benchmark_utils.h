#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                                 \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

namespace reduction {

struct Options {
    int n = 1 << 24;
    int warmup = 10;
    int iters = 50;
    bool check = false;
};

inline void PrintUsage(const char *prog) {
    std::printf("Usage: %s [--n N] [--warmup W] [--iters I] [--check]\n", prog);
}

inline bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    ++idx;
    return true;
}

inline bool ValidateBasicOptions(const Options &opt) {
    return opt.n > 0 && opt.warmup >= 0 && opt.iters > 0;
}

inline void InitRandom(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

inline double CpuReduce(const std::vector<float> &values) {
    double sum = 0.0;
    for (float v : values) {
        sum += static_cast<double>(v);
    }
    return sum;
}

inline double AbsDiff(double actual, double expected) {
    return std::fabs(actual - expected);
}

inline double RelDiff(double actual, double expected) {
    const double denom = std::max(std::fabs(expected), 1e-12);
    return std::fabs(actual - expected) / denom;
}

inline double ComputeBandwidthGbps(size_t bytes, float avg_ms) {
    return static_cast<double>(bytes) / (static_cast<double>(avg_ms) * 1.0e6);
}

}  // namespace reduction
