#include <cuda_runtime.h>

#include "transpose/benchmark_utils.h"
#include "transpose/kernels.h"

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

// 这是 transpose 的 baseline 版本：
// 每个线程直接从 input 读一个元素，然后写到转置后的 output。
//
// 它的访问模式有一个非常经典的问题：
// - 从 input 读时，x 方向通常是连续的，因此读比较友好
// - 写到 output 时，地址会跨 stride 跳跃，因此写不连续
//
// 这正是 transpose 在 CUDA 教学里常见的第一课：
// “同样是一次读写，访问方向不同，带宽利用率就会差很多。”
__global__ void transpose_naive(const float *input, float *output, int m, int n) {
    // x / y 是当前线程在输入矩阵中的逻辑坐标。
    const int x = blockIdx.x * kTileDim + threadIdx.x;
    const int y = blockIdx.y * kTileDim + threadIdx.y;

    // blockDim.y 只有 kBlockRows=8，
    // 因此一个线程会沿 y 方向额外处理 4 个元素，
    // 这样可以用较少线程覆盖一个 32x32 tile。
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = y + j;
        if (x < n && row < m) {
            // 输入坐标是 (row, x)，输出则要写成 (x, row)。
            output[x * m + row] = input[row * n + x];
        }
    }
}

// 一个 block 覆盖一个 32x32 的输入 tile。
void LaunchTranspose0(int m, int n, const float *input, float *output) {
    dim3 block(kTileDim, kBlockRows);
    dim3 grid((n + kTileDim - 1) / kTileDim, (m + kTileDim - 1) / kTileDim);
    transpose_naive<<<grid, block>>>(input, output, m, n);
}

// baseline 版本没有额外 shape 约束。
bool ValidateTranspose0(const transpose::Options &opt) {
    return transpose::ValidateBasicOptions(opt);
}

}  // namespace

namespace transpose {

// 返回这个 kernel 的说明书。
KernelSpec GetTranspose0Spec() {
    return KernelSpec{"transpose_0", LaunchTranspose0, ValidateTranspose0};
}

}  // namespace transpose
