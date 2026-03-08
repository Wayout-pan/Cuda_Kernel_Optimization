#pragma once

// 这个头文件放的是所有 kernel 都会复用的基础工具。
// 之所以单独抽出来，是为了避免每个 .cu 文件都重复写一遍：
// 1. 命令行参数结构
// 2. 参数解析
// 3. 输入合法性校验
// 4. CPU 参考实现
// 5. 简单的性能指标计算
//
// 对初学者来说，可以把它理解成：
// kernel 文件只关心“怎么在 GPU 上算”，
// 这个文件则关心“怎么把实验跑起来”。

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// CUDA API 基本都会返回一个 cudaError_t。
// 这个宏的作用是：如果调用失败，立即打印错误并退出程序。
// 这样可以避免错误被悄悄吞掉，便于调试。
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

// 这是一组 benchmark 运行参数。
// 对 GEMM 来说：
// - m: 输出矩阵 C 的行数，同时也是 A 的行数
// - n: 输出矩阵 C 的列数，同时也是 B 的列数
// - k: A 的列数，同时也是 B 的行数
//
// 如果写成矩阵维度：
// A 是 [m x k]
// B 是 [k x n]
// C 是 [m x n]
struct Options {
    int m = 4096;
    int n = 4096;
    int k = 4096;

    // warmup 表示预热次数。
    // GPU 第一次启动 kernel 往往会有额外开销，预热后再计时更稳定。
    int warmup = 10;

    // iters 表示正式测量时执行多少次。
    // 多跑几次再取平均值，结果更可靠。
    int iters = 50;

    // check=true 时，会使用 CPU 版本再算一遍，检查 GPU 结果是否正确。
    bool check = false;
};

// 打印帮助信息。
inline void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s [--m M] [--n N] [--k K] [--warmup W] [--iters I] [--check]\\n",
        prog);
}

// 解析形如 "--m 1024" 这样的整型参数。
// idx 会前进一位，因为它会顺手吃掉当前参数后面的数字。
inline bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    ++idx;
    return true;
}

// 通用命令行解析。
// 这里不直接依赖某个具体 kernel，因此 runner 和其他工具都可以复用。
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

// 基础合法性校验。
// 这里只检查值是不是正数，和 kernel 的具体 tile 约束无关。
inline bool ValidateBasicOptions(const Options &opt) {
    return opt.m > 0 && opt.n > 0 && opt.k > 0 && opt.warmup >= 0 && opt.iters > 0;
}

// 很多优化 kernel 会假设矩阵尺寸能被 tile 大小整除。
// 比如 block 一次处理 128x128 的输出块，那么 M 和 N 最好是 128 的倍数。
// 否则要额外写边界处理代码。
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

// 用固定随机数种子初始化输入，保证每次跑的数据一致，方便复现。
inline void InitRandom(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

// 计算两个结果向量的最大绝对误差。
// 对浮点数来说，GPU 和 CPU 结果通常不会逐位完全一样，
// 所以我们一般看误差是否在可接受范围内。
inline float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

// CPU 版本的 GEMM。
// 这是最直接、最容易理解的三重循环写法，也是最慢的写法之一。
// 它的主要作用不是快，而是作为“正确答案”参考。
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

// 求一个简单的 checksum。
// 它不能替代严格校验，但在快速比对时很有用：
// 如果两个 kernel 的 checksum 差很多，通常说明结果有问题。
inline double Checksum(const std::vector<float> &values) {
    double checksum = 0.0;
    for (float v : values) {
        checksum += v;
    }
    return checksum;
}

// GEMM 的理论浮点操作数近似是 2 * M * N * K。
// 为什么是 2？因为一次乘加通常按 1 次乘法 + 1 次加法计算。
// avg_ms 是毫秒，所以换算成 TFLOPS 时要除以 1e9。
inline double ComputeTflops(int m, int n, int k, float avg_ms) {
    return (2.0 * static_cast<double>(m) * n * k) / (static_cast<double>(avg_ms) * 1e9);
}

}  // namespace gemm
