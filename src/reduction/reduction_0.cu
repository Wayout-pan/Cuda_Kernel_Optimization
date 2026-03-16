#include <cuda_runtime.h>

#include "reduction/benchmark_utils.h"
#include "reduction/kernels.h"

namespace {

constexpr int kBlockSize = 256;

__global__ void reduce_single_block(const float *input, float *output, int n) {
    __shared__ float shared[kBlockSize];

    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int idx = tid; idx < n; idx += blockDim.x) {
        sum += input[idx];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = shared[0];
    }
}

void LaunchReduction0(int n, const float *input, float *output) {
    reduce_single_block<<<1, kBlockSize>>>(input, output, n);
}

bool ValidateReduction0(const reduction::Options &opt) {
    return reduction::ValidateBasicOptions(opt);
}

}  // namespace

namespace reduction {

KernelSpec GetReduction0Spec() {
    return KernelSpec{"reduction_0", LaunchReduction0, ValidateReduction0};
}

}  // namespace reduction
