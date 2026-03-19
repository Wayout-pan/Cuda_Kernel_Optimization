#include <cuda_runtime.h>

#include "reduction/benchmark_utils.h"
#include "reduction/kernels.h"

namespace {

constexpr int kBlockSize = 256;

// 这是最朴素的 reduction 写法：
// 整个向量只交给一个 block 来做。
//
// 执行过程可以分成两段：
// 1. 每个线程先在 global memory 上做 strided load，得到自己的局部和
// 2. 再把局部和写入 shared memory，做树形归约
//
// 这版的好处是逻辑最容易看清楚，
// 但缺点也很明显：只能使用一个 block，因此无法利用更大的 GPU 并行度。
__global__ void reduce_single_block(const float *input, float *output, int n) {
    __shared__ float shared[kBlockSize];

    const int tid = threadIdx.x;

    // 每个线程负责输入向量中一条 stride 序列：
    // tid, tid + blockDim.x, tid + 2 * blockDim.x, ...
    float sum = 0.0f;
    for (int idx = tid; idx < n; idx += blockDim.x) {
        sum += input[idx];
    }

    // 先把每个线程自己的局部和写进 shared memory。
    shared[tid] = sum;
    __syncthreads();

    // 再在 block 内做标准的树形归约。
    // stride 每次减半，直到只剩 shared[0]。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // 最终只有 thread 0 把 block 结果写回 global memory。
    if (tid == 0) {
        output[0] = shared[0];
    }
}

// baseline 版本只启动一个 block。
void LaunchReduction0(int n, const float *input, float *output) {
    reduce_single_block<<<1, kBlockSize>>>(input, output, n);
}

// 这版几乎没有 shape 限制，只要求参数本身合法。
bool ValidateReduction0(const reduction::Options &opt) {
    return reduction::ValidateBasicOptions(opt);
}

}  // namespace

namespace reduction {

// 返回这个 kernel 的说明书，供 registry 收集。
KernelSpec GetReduction0Spec() {
    return KernelSpec{"reduction_0", LaunchReduction0, ValidateReduction0};
}

}  // namespace reduction
