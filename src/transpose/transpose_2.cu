#include <cuda_runtime.h>

#include "transpose/benchmark_utils.h"
#include "transpose/kernels.h"

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

// 在 transpose_1 的基础上，shared memory 的第二维多开 1。
// 经典作用是打破 bank 对齐冲突，让转置读写时不同线程更少撞到同一个 bank。
__global__ void transpose_shared_padded(const float *input, float *output, int m, int n) {
    // 和 transpose_1 的唯一区别就在这里：
    // 第二维从 32 变成了 33。
    //
    // 多出来的 1 列不会改变逻辑上的 tile 大小，
    // 但会改变 shared memory 中相邻行的起始地址对齐方式，
    // 从而打破 bank conflict。
    __shared__ float tile[kTileDim][kTileDim + 1];

    const int x = blockIdx.x * kTileDim + threadIdx.x;
    const int y = blockIdx.y * kTileDim + threadIdx.y;

    // 第一阶段：先按原布局装载输入 tile。
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = y + j;
        if (x < n && row < m) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * n + x];
        }
    }

    __syncthreads();

    // 第二阶段：交换 block 在输出矩阵中的位置，和 transpose_1 一样。
    const int transposed_x = blockIdx.y * kTileDim + threadIdx.x;
    const int transposed_y = blockIdx.x * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = transposed_y + j;
        if (transposed_x < m && row < n) {
            // 逻辑仍然是转置读取，但因为 shared memory 布局做了 padding，
            // 这一步的 bank conflict 会明显少很多。
            output[row * m + transposed_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// launch 形状仍然保持一致，便于直接比较优化收益。
void LaunchTranspose2(int m, int n, const float *input, float *output) {
    dim3 block(kTileDim, kBlockRows);
    dim3 grid((n + kTileDim - 1) / kTileDim, (m + kTileDim - 1) / kTileDim);
    transpose_shared_padded<<<grid, block>>>(input, output, m, n);
}

bool ValidateTranspose2(const transpose::Options &opt) {
    return transpose::ValidateBasicOptions(opt);
}

}  // namespace

namespace transpose {

KernelSpec GetTranspose2Spec() {
    return KernelSpec{"transpose_2", LaunchTranspose2, ValidateTranspose2};
}

}  // namespace transpose
