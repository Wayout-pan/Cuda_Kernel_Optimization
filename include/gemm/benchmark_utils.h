#pragma once

#include <cuda_runtime.h>

#include <algorithm>
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
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                                 \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

namespace gemm {

struct Options {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 10;
    int iters = 50;
    bool check = false;
};

inline void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s [--m M] [--n N] [--k K] [--warmup W] [--iters I] [--check]\\n",
        prog);
}

inline bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    ++idx;
    return true;
}

inline bool ParseArgs(int argc, char **argv, Options &opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--m") {
            if (!ParseIntArg(argc, argv, i, opt.m)) return false;
        } else if (arg == "--n") {
            if (!ParseIntArg(argc, argv, i, opt.n)) return false;
        } else if (arg == "--k") {
            if (!ParseIntArg(argc, argv, i, opt.k)) return false;
        } else if (arg == "--warmup") {
            if (!ParseIntArg(argc, argv, i, opt.warmup)) return false;
        } else if (arg == "--iters") {
            if (!ParseIntArg(argc, argv, i, opt.iters)) return false;
        } else if (arg == "--check") {
            opt.check = true;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            return false;
        }
    }
    return true;
}

inline bool ValidateBasicOptions(const Options &opt) {
    return opt.m > 0 && opt.n > 0 && opt.k > 0 && opt.warmup >= 0 && opt.iters > 0;
}

inline bool ValidateDivisibility(const Options &opt, int tile_m, int tile_n, int tile_k) {
    if (!ValidateBasicOptions(opt)) {
        return false;
    }
    if (opt.m % tile_m != 0 || opt.n % tile_n != 0 || opt.k % tile_k != 0) {
        std::fprintf(stderr,
                     "Shape constraints not satisfied: require M %% %d == 0, N %% %d == 0, K %% %d == 0.\\n",
                     tile_m,
                     tile_n,
                     tile_k);
        return false;
    }
    return true;
}

inline void InitRandom(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

inline float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

inline void CpuGemm(const std::vector<float> &a,
                    const std::vector<float> &b,
                    std::vector<float> &c,
                    int m,
                    int n,
                    int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

inline double Checksum(const std::vector<float> &values) {
    double checksum = 0.0;
    for (float v : values) {
        checksum += v;
    }
    return checksum;
}

inline double ComputeTflops(int m, int n, int k, float avg_ms) {
    return (2.0 * static_cast<double>(m) * n * k) / (static_cast<double>(avg_ms) * 1e9);
}

}  // namespace gemm
