#include <cuda_runtime.h>

#include <algorithm>

#include "reduction/benchmark_utils.h"
#include "reduction/kernels.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;

__global__ void reduce_atomic(const float *input, float *output, int n) {
    __shared__ float shared[kBlockSize];

    const int tid = threadIdx.x;
    const unsigned long long base = static_cast<unsigned long long>(blockIdx.x) * blockDim.x * 2 + tid;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x * 2;

    float sum = 0.0f;
    for (unsigned long long idx = base; idx < static_cast<unsigned long long>(n); idx += stride) {
        sum += input[idx];
        const unsigned long long pair_idx = idx + blockDim.x;
        if (pair_idx < static_cast<unsigned long long>(n)) {
            sum += input[pair_idx];
        }
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride_step = blockDim.x / 2; stride_step >= 32; stride_step >>= 1) {
        if (tid < stride_step) {
            shared[tid] += shared[tid + stride_step];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *warp_shared = shared;
        warp_shared[tid] += warp_shared[tid + 32];
        warp_shared[tid] += warp_shared[tid + 16];
        warp_shared[tid] += warp_shared[tid + 8];
        warp_shared[tid] += warp_shared[tid + 4];
        warp_shared[tid] += warp_shared[tid + 2];
        warp_shared[tid] += warp_shared[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

void LaunchReduction1(int n, const float *input, float *output) {
    const int blocks = std::max(1, std::min((n + kBlockSize * 2 - 1) / (kBlockSize * 2), kMaxBlocks));
    reduce_atomic<<<blocks, kBlockSize>>>(input, output, n);
}

bool ValidateReduction1(const reduction::Options &opt) {
    return reduction::ValidateBasicOptions(opt);
}

}  // namespace

namespace reduction {

KernelSpec GetReduction1Spec() {
    return KernelSpec{"reduction_1", LaunchReduction1, ValidateReduction1};
}

}  // namespace reduction
