#include <cuda_runtime.h>

#include <algorithm>

#include "reduction/benchmark_utils.h"
#include "reduction/kernels.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;

// 这一版把 reduction 拆给多个 block 并行执行。
// 每个 block 先在自己的范围内完成局部归约，
// 最后 block 之间通过 atomicAdd 聚合到同一个输出地址。
//
// 相比 reduction_0，它的核心提升是：
// 1. 能利用多个 block 的并行度
// 2. 每个线程一次读取两个元素，减少循环次数
// 3. 最后 32 个线程用 warp-synchronous 的方式收尾，减少同步开销
__global__ void reduce_atomic(const float *input, float *output, int n) {
    __shared__ float shared[kBlockSize];

    const int tid = threadIdx.x;

    // base 是“当前线程第一次该读哪里”。
    // 一个 block 一轮最多覆盖 2 * blockDim.x 个元素，
    // 因此 blockIdx.x 会先乘上这个跨度。
    const unsigned long long base = static_cast<unsigned long long>(blockIdx.x) * blockDim.x * 2 + tid;

    // stride 决定“下一轮该跳多远”。
    // 因为整个 grid 中每个 block 都按 2 * blockDim.x 的粒度前进，
    // 所以 stride 也要乘上 gridDim.x。
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x * 2;

    float sum = 0.0f;
    for (unsigned long long idx = base; idx < static_cast<unsigned long long>(n); idx += stride) {
        sum += input[idx];

        // 每个线程顺手再读一个“配对元素”，
        // 这样一个 warp 一轮就能多覆盖一倍数据。
        const unsigned long long pair_idx = idx + blockDim.x;
        if (pair_idx < static_cast<unsigned long long>(n)) {
            sum += input[pair_idx];
        }
    }

    // 和 reduction_0 一样，先把局部和写进 shared memory。
    shared[tid] = sum;
    __syncthreads();

    // 先把 block 内前半段的树形归约做完，直到只剩一个 warp。
    for (int stride_step = blockDim.x / 2; stride_step >= 32; stride_step >>= 1) {
        if (tid < stride_step) {
            shared[tid] += shared[tid + stride_step];
        }
        __syncthreads();
    }

    // 剩下 32 个线程时，就进入 warp-synchronous 收尾。
    // 这里用 volatile 指针，是经典 CUDA reduction 教学写法。
    if (tid < 32) {
        volatile float *warp_shared = shared;
        warp_shared[tid] += warp_shared[tid + 32];
        warp_shared[tid] += warp_shared[tid + 16];
        warp_shared[tid] += warp_shared[tid + 8];
        warp_shared[tid] += warp_shared[tid + 4];
        warp_shared[tid] += warp_shared[tid + 2];
        warp_shared[tid] += warp_shared[tid + 1];
    }

    // 每个 block 的 thread 0 用 atomicAdd 把局部结果汇总到 output。
    // 这会带来一定原子操作开销，但实现简单，也适合教学示例。
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// blocks 数量会随 n 增长，但不会超过 kMaxBlocks。
// 这是一个“简单够用”的策略，目的是避免 block 数无限膨胀。
void LaunchReduction1(int n, const float *input, float *output) {
    const int blocks = std::max(1, std::min((n + kBlockSize * 2 - 1) / (kBlockSize * 2), kMaxBlocks));
    reduce_atomic<<<blocks, kBlockSize>>>(input, output, n);
}

// 当前实现没有额外 shape 约束。
bool ValidateReduction1(const reduction::Options &opt) {
    return reduction::ValidateBasicOptions(opt);
}

}  // namespace

namespace reduction {

KernelSpec GetReduction1Spec() {
    return KernelSpec{"reduction_1", LaunchReduction1, ValidateReduction1};
}

}  // namespace reduction
