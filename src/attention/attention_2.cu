#include <cuda_runtime.h>

#include <algorithm>

#include "attention/benchmark_utils.h"
#include "attention/kernels.h"

namespace {

constexpr int kMaxHeadDim = 128;
constexpr int kMaxThreads = 128;
constexpr int kTileKeys = 8;
constexpr float kNegativeInfinity = -1.0e20f;

// 这一版在 attention_1 的基础上，再引入 key tile。
// 每轮先把一小块 K/V 搬到 shared memory，
// 再在 block 内复用这些数据，减少对 global memory 的重复访问。
//
// 它仍然保持 fused + online softmax 的整体思路，
// 但在访存层面更进一步。
__global__ void attention_tiled_online(const float *q,
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

    // shared memory 被切成三段：
    // 1. k_tile: 当前 key tile
    // 2. v_tile: 当前 value tile
    // 3. reduction: 当前点积归约缓冲区
    extern __shared__ float shared[];
    float *k_tile = shared;
    float *v_tile = k_tile + kTileKeys * head_dim;
    float *reduction = v_tile + kTileKeys * head_dim;

    const int tid = threadIdx.x;
    const int bh = row_idx / seq_len;
    const int row = row_idx % seq_len;
    const size_t row_offset = attention::Offset(bh, row, 0, seq_len, head_dim);

    __shared__ float alpha_shared;
    __shared__ float prob_shared;
    __shared__ float row_sum_shared;

    const float q_value = tid < head_dim ? q[row_offset + tid] : 0.0f;
    float accum = 0.0f;
    float running_max = kNegativeInfinity;
    float running_sum = 0.0f;

    // 外层循环按 key tile 推进，而不是一次只处理一个 key row。
    for (int tile_start = 0; tile_start < seq_len; tile_start += kTileKeys) {
        const int valid_keys = min(kTileKeys, seq_len - tile_start);

        // 先把当前 tile 的 K/V 搬进 shared memory。
        for (int idx = tid; idx < valid_keys * head_dim; idx += blockDim.x) {
            const int local_key = idx / head_dim;
            const int d = idx % head_dim;
            const size_t kv_offset = attention::Offset(bh, tile_start + local_key, d, seq_len, head_dim);
            k_tile[idx] = k[kv_offset];
            v_tile[idx] = v[kv_offset];
        }
        __syncthreads();

        // 然后在 shared memory 上逐行处理当前 tile 内的 key。
        for (int local_key = 0; local_key < valid_keys; ++local_key) {
            // block 内先完成 q_row 和当前 k_row 的点积。
            reduction[tid] = tid < head_dim ? q_value * k_tile[local_key * head_dim + tid] : 0.0f;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduction[tid] += reduction[tid + stride];
                }
                __syncthreads();
            }

            // thread 0 更新 online softmax 状态。
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

            // 所有线程一起更新输出向量的各个维度。
            if (tid < head_dim) {
                accum = accum * alpha_shared + prob_shared * v_tile[local_key * head_dim + tid];
            }
            __syncthreads();
        }
    }

    if (tid < head_dim) {
        o[row_offset + tid] = accum / row_sum_shared;
    }
}

// 选一个不小于 head_dim 的 2 次幂线程数。
int NextPow2(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

// shared_bytes 由三部分组成：
// - kTileKeys * head_dim 个 K 元素
// - kTileKeys * head_dim 个 V 元素
// - threads 个 reduction 缓冲区元素
void LaunchAttention2(int batch,
                      int heads,
                      int seq_len,
                      int head_dim,
                      const float *q,
                      const float *k,
                      const float *v,
                      float *o) {
    const int total_rows = batch * heads * seq_len;
    const int threads = std::min(kMaxThreads, NextPow2(std::max(32, head_dim)));
    const size_t shared_bytes =
        static_cast<size_t>(2 * kTileKeys * head_dim + threads) * sizeof(float);
    attention_tiled_online<<<total_rows, threads, shared_bytes>>>(
        q, k, v, o, total_rows, seq_len, head_dim, static_cast<float>(attention::AttentionScale(head_dim)));
}

bool ValidateAttention2(const attention::Options &opt) {
    return attention::ValidateHeadDim(opt, kMaxHeadDim);
}

}  // namespace

namespace attention {

KernelSpec GetAttention2Spec() {
    return KernelSpec{"attention_2", LaunchAttention2, ValidateAttention2};
}

}  // namespace attention
