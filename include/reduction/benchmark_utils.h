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

// reduction benchmark 的公共参数。
// 这里的 n 表示输入向量长度，也就是要被求和的元素个数。
struct Options {
    int n = 1 << 24;
    int warmup = 10;
    int iters = 50;
    bool check = false;
};

// 打印帮助信息。
inline void PrintUsage(const char *prog) {
    std::printf("Usage: %s [--n N] [--warmup W] [--iters I] [--check]\n", prog);
}

// 解析形如 "--n 1048576" 这样的整型参数。
inline bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    ++idx;
    return true;
}

// reduction 的基础参数校验。
// 这里只关心长度、warmup 和迭代次数是不是合法，
// 不去掺杂某个具体 kernel 的 block/grid 约束。
inline bool ValidateBasicOptions(const Options &opt) {
    return opt.n > 0 && opt.warmup >= 0 && opt.iters > 0;
}

// 用固定随机数种子初始化输入，保证 benchmark 可复现。
inline void InitRandom(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

// CPU 版本的 sum reduction。
// 这里故意用最直接的串行写法，因为它的目标是提供“正确答案”，
// 而不是追求和 GPU 一样的性能。
inline double CpuReduce(const std::vector<float> &values) {
    double sum = 0.0;
    for (float v : values) {
        sum += static_cast<double>(v);
    }
    return sum;
}

// 绝对误差。
inline double AbsDiff(double actual, double expected) {
    return std::fabs(actual - expected);
}

// 相对误差。
// expected 很小时要避免分母接近 0，因此这里做了一个最小值保护。
inline double RelDiff(double actual, double expected) {
    const double denom = std::max(std::fabs(expected), 1e-12);
    return std::fabs(actual - expected) / denom;
}

// reduction 更像 memory-bound 算子，
// 所以这里更适合用带宽而不是 FLOPS 来度量。
// bytes 通常就是“读取输入向量的字节数”。
inline double ComputeBandwidthGbps(size_t bytes, float avg_ms) {
    return static_cast<double>(bytes) / (static_cast<double>(avg_ms) * 1.0e6);
}

}  // namespace reduction
