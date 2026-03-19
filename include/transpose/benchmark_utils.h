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
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                                 \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

namespace transpose {

// transpose benchmark 的公共参数。
// 这里的 m / n 表示输入矩阵 A 的 shape = [m x n]，
// 输出矩阵 B 的 shape 则是 [n x m]。
struct Options {
    int m = 4096;
    int n = 4096;
    int warmup = 10;
    int iters = 50;
    bool check = false;
};

// 打印帮助信息。
inline void PrintUsage(const char *prog) {
    std::printf("Usage: %s [--m M] [--n N] [--warmup W] [--iters I] [--check]\n", prog);
}

// 解析形如 "--m 4096" 这样的整型参数。
inline bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    ++idx;
    return true;
}

// transpose 的基础合法性校验。
inline bool ValidateBasicOptions(const Options &opt) {
    return opt.m > 0 && opt.n > 0 && opt.warmup >= 0 && opt.iters > 0;
}

// 用固定随机种子初始化输入，便于复现实验结果。
inline void InitRandom(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

// CPU 参考版 transpose。
// 最直接的定义就是：output[col, row] = input[row, col]。
inline void CpuTranspose(const std::vector<float> &input, std::vector<float> &output, int m, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            output[col * m + row] = input[row * n + col];
        }
    }
}

// 计算 GPU 输出和 CPU 参考输出之间的最大绝对误差。
inline float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

// 简单 checksum，用于快速对比结果。
inline double Checksum(const std::vector<float> &values) {
    double checksum = 0.0;
    for (float v : values) {
        checksum += v;
    }
    return checksum;
}

// transpose 同样偏 memory-bound。
// 一次转置大致会经历一次读 + 一次写，因此这里乘以 2。
inline double ComputeBandwidthGbps(size_t element_count, float avg_ms) {
    const double bytes = static_cast<double>(element_count) * sizeof(float) * 2.0;
    return bytes / (static_cast<double>(avg_ms) * 1.0e6);
}

}  // namespace transpose
