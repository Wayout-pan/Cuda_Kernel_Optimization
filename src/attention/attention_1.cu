#include <cuda_runtime.h>

#include "attention/benchmark_utils.h"
#include "attention/kernels.h"

namespace {

constexpr int kMaxHeadDim = 128;
constexpr int kMaxThreads = 128;
constexpr float kNegativeInfinity = -1.0e20f;

// 这一版开始把一个 query row 交给“整个 block”来做。
// block 内线程会合作完成两件事：
// 1. 计算 q_row 和每个 k_row 的点积
// 2. 以 online softmax 的方式一边扫描 key，一边更新输出
//
// 它比 attention_0 更接近现代 attention 优化思路：
// 不显式存整行 score，而是边算 score、边更新 softmax 状态、边累积输出。
__global__ void attention_fused_online(const float *q,
                                       const float *k,
                                       const float *v,
                                       float *o,
                                       int total_rows,
                                       int seq_len,
                                       int head_dim,
                                       float scale) {
    const int row_idx = blockIdx.x;
    if (row_idx >= total_rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int bh = row_idx / seq_len;
    const int row = row_idx % seq_len;
    const size_t row_offset = attention::Offset(bh, row, 0, seq_len, head_dim);

    // reduction 用来做 q_row 和 k_row 的点积归约。
    __shared__ float reduction[kMaxThreads];

    // alpha_shared / prob_shared / row_sum_shared
    // 是 online softmax 这一轮需要共享给整个 block 的状态。
    __shared__ float alpha_shared;
    __shared__ float prob_shared;
    __shared__ float row_sum_shared;

    // 每个线程只持有 q_row 中的一个元素。
    const float q_value = tid < head_dim ? q[row_offset + tid] : 0.0f;
    float accum = 0.0f;
    float running_max = kNegativeInfinity;
    float running_sum = 0.0f;

    // 外层循环按 key row 扫描整条序列。
    for (int col = 0; col < seq_len; ++col) {
        const size_t kv_offset = attention::Offset(bh, col, 0, seq_len, head_dim);

        // 第一步：每个线程贡献一个乘积，block 内做点积归约。
        reduction[tid] = tid < head_dim ? q_value * k[kv_offset + tid] : 0.0f;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        // 第二步：thread 0 用点积结果更新 online softmax 状态。
        if (tid == 0) {
            const float score = reduction[0] * scale;
            const float new_max = fmaxf(running_max, score);
            alpha_shared = expf(running_max - new_max);
            prob_shared = expf(score - new_max);
            running_sum = running_sum * alpha_shared + prob_shared;
            running_max = new_max;
            row_sum_shared = running_sum;
        }
        __syncthreads();

        // 第三步：每个线程负责输出向量中的一个维度，
        // 按 online softmax 的更新规则维护 accum。
        if (tid < head_dim) {
            accum = accum * alpha_shared + prob_shared * v[kv_offset + tid];
        }
        __syncthreads();
    }

    // 全部 key 扫描结束后，除以最终分母得到输出。
    if (tid < head_dim) {
        o[row_offset + tid] = accum / row_sum_shared;
    }
}

// 选一个不小于 head_dim 的 2 次幂线程数，便于做树形归约。
int NextPow2(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

// 一个 block 对应一个 query row。
void LaunchAttention1(int batch,
                      int heads,
                      int seq_len,
                      int head_dim,
                      const float *q,
                      const float *k,
                      const float *v,
                      float *o) {
    const int total_rows = batch * heads * seq_len;
    const int threads = std::min(kMaxThreads, NextPow2(std::max(32, head_dim)));
    attention_fused_online<<<total_rows, threads>>>(
        q, k, v, o, total_rows, seq_len, head_dim, static_cast<float>(attention::AttentionScale(head_dim)));
}

bool ValidateAttention1(const attention::Options &opt) {
    return attention::ValidateHeadDim(opt, kMaxHeadDim);
}

}  // namespace

namespace attention {

KernelSpec GetAttention1Spec() {
    return KernelSpec{"attention_1", LaunchAttention1, ValidateAttention1};
}

}  // namespace attention
