#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

namespace attention {

// attention benchmark 的公共参数。
// 这里采用大模型里常见的四元组：
// - batch: batch size
// - heads: attention heads 数量
// - seq_len: 序列长度
// - head_dim: 每个 head 的隐藏维度
struct Options {
    int batch = 1;
    int heads = 8;
    int seq_len = 128;
    int head_dim = 64;
    int warmup = 10;
    int iters = 50;
    bool check = false;
};

// 打印帮助信息。
inline void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s [--batch B] [--heads H] [--seq-len N] [--head-dim D] [--warmup W] [--iters I] [--check]\n",
        prog);
}

// 解析整型命令行参数。
inline bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    ++idx;
    return true;
}

// attention 的基础合法性校验。
inline bool ValidateBasicOptions(const Options &opt) {
    return opt.batch > 0 && opt.heads > 0 && opt.seq_len > 0 && opt.head_dim > 0 &&
           opt.warmup >= 0 && opt.iters > 0;
}

// 某些教学版 kernel 会把 head_dim 的上界写死在寄存器数组大小里，
// 因此这里提供一个额外的 head_dim 约束检查。
inline bool ValidateHeadDim(const Options &opt, int max_head_dim) {
    if (!ValidateBasicOptions(opt)) {
        return false;
    }
    if (opt.head_dim > max_head_dim) {
        std::fprintf(stderr,
                     "Head dim constraint not satisfied: require D <= %d, got %d.\n",
                     max_head_dim,
                     opt.head_dim);
        return false;
    }
    return true;
}

// 用固定随机种子初始化 Q/K/V，保证实验可复现。
inline void InitRandom(std::vector<float> &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : values) {
        v = dist(rng);
    }
}

// 计算 GPU 输出与 CPU 参考输出之间的最大绝对误差。
inline float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

// 简单 checksum，用于快速检查不同 kernel 输出是否大体一致。
inline double Checksum(const std::vector<float> &values) {
    double checksum = 0.0;
    for (float v : values) {
        checksum += v;
    }
    return checksum;
}

// attention 的缩放因子，来源于标准 scaled dot-product attention 定义。
inline double AttentionScale(int head_dim) {
    return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

// 这里给一个近似 FLOPs：
// - QK^T 近似 2 * B * H * N * N * D
// - P * V 近似 2 * B * H * N * N * D
// 合起来近似 4 * B * H * N * N * D
inline long long ApproxFlops(const Options &opt) {
    return 4LL * opt.batch * opt.heads * opt.seq_len * opt.seq_len * opt.head_dim;
}

// 根据近似 FLOPs 计算 TFLOPS。
inline double ComputeTflops(const Options &opt, float avg_ms) {
    return static_cast<double>(ApproxFlops(opt)) / (static_cast<double>(avg_ms) * 1.0e9);
}

// 统一的线性地址函数。
// bh 把 batch 和 head 合并成一个维度，方便 runner 和 kernel 共用。
__host__ __device__ inline size_t Offset(int bh, int row, int col, int seq_len, int head_dim) {
    return (static_cast<size_t>(bh) * seq_len + row) * head_dim + col;
}

// CPU 参考版 attention。
// 这里按最直观的三步来做：
// 1. 计算一整行 score
// 2. 做 softmax
// 3. 用 softmax 概率加权 V，得到输出向量
inline void CpuAttention(const std::vector<float> &q,
                         const std::vector<float> &k,
                         const std::vector<float> &v,
                         std::vector<float> &o,
                         int batch,
                         int heads,
                         int seq_len,
                         int head_dim) {
    const int total_heads = batch * heads;
    const float scale = static_cast<float>(AttentionScale(head_dim));

    // scores 保存“当前 query row 对所有 key row 的分数”。
    std::vector<float> scores(seq_len, 0.0f);

    for (int bh = 0; bh < total_heads; ++bh) {
        for (int row = 0; row < seq_len; ++row) {
            // softmax 前先找 row max，用于数值稳定。
            float row_max = -std::numeric_limits<float>::infinity();
            for (int col = 0; col < seq_len; ++col) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q[Offset(bh, row, d, seq_len, head_dim)] *
                             k[Offset(bh, col, d, seq_len, head_dim)];
                }
                score *= scale;
                scores[col] = score;
                row_max = std::max(row_max, score);
            }

            // 先清空这一行的输出。
            float denom = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                o[Offset(bh, row, d, seq_len, head_dim)] = 0.0f;
            }

            // 再做 exp / sum，并顺手累计概率加权后的 V。
            for (int col = 0; col < seq_len; ++col) {
                const float prob = std::exp(scores[col] - row_max);
                denom += prob;
                for (int d = 0; d < head_dim; ++d) {
                    o[Offset(bh, row, d, seq_len, head_dim)] +=
                        prob * v[Offset(bh, col, d, seq_len, head_dim)];
                }
            }

            const float inv = 1.0f / denom;

            // 最后统一除以 softmax 分母。
            for (int d = 0; d < head_dim; ++d) {
                o[Offset(bh, row, d, seq_len, head_dim)] *= inv;
            }
        }
    }
}

}  // namespace attention
