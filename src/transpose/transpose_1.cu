#include <cuda_runtime.h>

#include "transpose/benchmark_utils.h"
#include "transpose/kernels.h"

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

// 这一版先把 tile 搬到 shared memory，再由 block 内线程合作写回。
// 这样全局读和全局写都能保持连续，但 shared memory 转置访问仍会有 bank conflict。
__global__ void transpose_shared(const float *input, float *output, int m, int n) {
    // tile 的逻辑 shape 是 32x32。
    // 先按输入布局写进 shared memory，
    // 再在写回时交换行列角色。
    __shared__ float tile[kTileDim][kTileDim];

    const int x = blockIdx.x * kTileDim + threadIdx.x;
    const int y = blockIdx.y * kTileDim + threadIdx.y;

    // 第一阶段：把输入矩阵的一个 tile 搬到 shared memory。
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = y + j;
        if (x < n && row < m) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * n + x];
        }
    }

    // 必须等整个 tile 都装完，后面才能开始按转置方式读取。
    __syncthreads();

    // 第二阶段：交换 block 在输出矩阵中的角色。
    // 原来 blockIdx.x 对应输入的列块，
    // 现在写回输出时要变成行块。
    const int transposed_x = blockIdx.y * kTileDim + threadIdx.x;
    const int transposed_y = blockIdx.x * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = transposed_y + j;
        if (transposed_x < m && row < n) {
            // 这里的读取 tile[threadIdx.x][threadIdx.y + j]
            // 就相当于在 shared memory 中做了一次“行列交换”。
            output[row * m + transposed_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// launch 形状和 baseline 一致，便于直接做性能对比。
void LaunchTranspose1(int m, int n, const float *input, float *output) {
    dim3 block(kTileDim, kBlockRows);
    dim3 grid((n + kTileDim - 1) / kTileDim, (m + kTileDim - 1) / kTileDim);
    transpose_shared<<<grid, block>>>(input, output, m, n);
}

// 这版没有强制要求 m/n 必须是 tile 的整数倍，
// 因为 kernel 里已经做了边界判断。
bool ValidateTranspose1(const transpose::Options &opt) {
    return transpose::ValidateBasicOptions(opt);
}

}  // namespace

namespace transpose {

KernelSpec GetTranspose1Spec() {
    return KernelSpec{"transpose_1", LaunchTranspose1, ValidateTranspose1};
}

}  // namespace transpose
