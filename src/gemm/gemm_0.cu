#include <cuda_runtime.h>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

namespace {

__global__ void sgemm_naive(const float *a,
                            const float *b,
                            float *c,
                            int m,
                            int n,
                            int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += a[row * k + kk] * b[kk * n + col];
    }
    c[row * n + col] = acc;
}

void LaunchGemm0(int m, int n, int k, const float *a, const float *b, float *c) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    sgemm_naive<<<grid, block>>>(a, b, c, m, n, k);
}

bool ValidateGemm0(const gemm::Options &opt) {
    return gemm::ValidateBasicOptions(opt);
}

}  // namespace

namespace gemm {

KernelSpec GetGemm0Spec() {
    return KernelSpec{"gemm_0", LaunchGemm0, ValidateGemm0};
}

}  // namespace gemm
