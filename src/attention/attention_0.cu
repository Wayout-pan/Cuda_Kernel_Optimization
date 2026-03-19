#include <cuda_runtime.h>

#include "attention/benchmark_utils.h"
#include "attention/kernels.h"

namespace {

constexpr int kMaxHeadDim = 128;
constexpr float kNegativeInfinity = -1.0e20f;

// baseline: 一个线程负责一个 query row 的完整输出向量。
// 这版几乎没有并行化 softmax 或输出维度，因此非常慢，但逻辑最直接。
__global__ void attention_naive_forward(const float *q,
                                        const float *k,
                                        const float *v,
                                        float *o,
                                        int total_rows,
                                        int seq_len,
                                        int head_dim,
                                        float scale) {
    // row_idx 把所有 (batch, head, query_row) 合并成一个一维编号。
    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= total_rows) {
        return;
    }

    // bh 表示“第几个 batch-head 组合”，
    // row 表示这个 head 内的第几个 query token。
    const int bh = row_idx / seq_len;
    const int row = row_idx % seq_len;
    const float *q_row = q + attention::Offset(bh, row, 0, seq_len, head_dim);
    float *o_row = o + attention::Offset(bh, row, 0, seq_len, head_dim);

    // 第一遍扫描：计算这一行所有 score 的最大值，供 softmax 做数值稳定。
    float max_score = kNegativeInfinity;
    for (int col = 0; col < seq_len; ++col) {
        const float *k_row = k + attention::Offset(bh, col, 0, seq_len, head_dim);
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_row[d] * k_row[d];
        }
        max_score = fmaxf(max_score, score * scale);
    }

    // accum 放在寄存器里，保存当前输出向量。
    // 这也是为什么这版会限制 head_dim <= 128。
    float accum[kMaxHeadDim] = {0.0f};
    float denom = 0.0f;

    // 第二遍扫描：
    // 1. 重新计算 score
    // 2. 做 exp(score - max)
    // 3. 累计 softmax 分母
    // 4. 顺手做 prob * V 的向量累加
    for (int col = 0; col < seq_len; ++col) {
        const float *k_row = k + attention::Offset(bh, col, 0, seq_len, head_dim);
        const float *v_row = v + attention::Offset(bh, col, 0, seq_len, head_dim);
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_row[d] * k_row[d];
        }
        const float prob = expf(score * scale - max_score);
        denom += prob;
        for (int d = 0; d < head_dim; ++d) {
            accum[d] += prob * v_row[d];
        }
    }

    // 最后统一除以 softmax 分母，得到最终输出。
    const float inv = 1.0f / denom;
    for (int d = 0; d < head_dim; ++d) {
        o_row[d] = accum[d] * inv;
    }
}

// baseline 采用普通的一维 grid，threads 只是在“行粒度”上并行。
void LaunchAttention0(int batch,
                      int heads,
                      int seq_len,
                      int head_dim,
                      const float *q,
                      const float *k,
                      const float *v,
                      float *o) {
    const int total_rows = batch * heads * seq_len;
    constexpr int kThreads = 128;
    const int blocks = (total_rows + kThreads - 1) / kThreads;
    attention_naive_forward<<<blocks, kThreads>>>(
        q, k, v, o, total_rows, seq_len, head_dim, static_cast<float>(attention::AttentionScale(head_dim)));
}

// 这版因为把输出向量放在寄存器数组里，所以限制了 head_dim 上界。
bool ValidateAttention0(const attention::Options &opt) {
    return attention::ValidateHeadDim(opt, kMaxHeadDim);
}

}  // namespace

namespace attention {

KernelSpec GetAttention0Spec() {
    return KernelSpec{"attention_0", LaunchAttention0, ValidateAttention0};
}

}  // namespace attention
