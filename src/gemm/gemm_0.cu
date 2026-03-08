#include <cuda_runtime.h>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

namespace {

// 这是最朴素的 GEMM 写法：
// 一个线程(thread)只负责输出矩阵 C 中的一个元素。
//
// 对应关系是：
// - row 表示 C 的行号
// - col 表示 C 的列号
// - 该线程会计算 C[row, col]
//
// 这是最容易理解的 CUDA GEMM 入门版本，但性能通常不高，原因包括：
// 1. 没有使用 shared memory 复用数据
// 2. 对 A/B 的访问会产生很多重复的 global memory 读
// 3. 每个线程做的工作量较小，访存压力较大
__global__ void sgemm_naive(const float *a,
                            const float *b,
                            float *c,
                            int m,
                            int n,
                            int k) {
    // blockIdx / threadIdx 是 CUDA 内建变量：
    // - blockIdx 表示当前 block 在整个 grid 中的位置
    // - threadIdx 表示当前线程在 block 中的位置
    // - blockDim 表示一个 block 的尺寸
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 对不能整除的尺寸，边界线程可能会落在矩阵外面，因此要保护一下。
    if (row >= m || col >= n) {
        return;
    }

    // 计算 C[row, col] = sum_k A[row, kk] * B[kk, col]
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += a[row * k + kk] * b[kk * n + col];
    }
    c[row * n + col] = acc;
}

// 这是给 runner 调用的统一 launch 函数。
// 这里决定 block/grid 的大小，但不负责分配显存或计时。
void LaunchGemm0(int m, int n, int k, const float *a, const float *b, float *c) {
    // 一个 16x16 的 block 一共有 256 个线程。
    // 这是 CUDA 教学示例里比较经典的尺寸。
    dim3 block(16, 16);

    // grid 决定一共需要多少个 block 才能覆盖整个输出矩阵。
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    sgemm_naive<<<grid, block>>>(a, b, c, m, n, k);
}

// baseline 版本几乎没有特殊尺寸要求，只要参数本身是正数即可。
bool ValidateGemm0(const gemm::Options &opt) {
    return gemm::ValidateBasicOptions(opt);
}

}  // namespace

namespace gemm {

// 返回这个 kernel 的“说明书”，供 registry 注册。
KernelSpec GetGemm0Spec() {
    return KernelSpec{"gemm_0", LaunchGemm0, ValidateGemm0};
}

}  // namespace gemm
