#include <cuda_runtime.h>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

// 把二维下标映射到一维线性地址。
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

namespace {

// 这一版比 gemm_1 更激进，核心思想是：
// 1. 仍然采用 tiled GEMM
// 2. 进一步强化寄存器 blocking
// 3. 让一个 warp 内线程以更细致的方式分工计算 C 子块
//
// 这类代码的可读性会明显下降，但通常换来更好的性能上限。
constexpr int kBlockSize = 16;  // block 维度为 16x16，共 256 个线程
constexpr int kBlockM = 128;    // block 负责的输出块行数
constexpr int kBlockN = 128;    // block 负责的输出块列数
constexpr int kBlockK = 8;      // 每次沿 K 维推进的深度
constexpr int kTm = kBlockM / kBlockSize;
constexpr int kTn = kBlockN / kBlockSize;

__global__ void matrixMul(const float *A,
                          const float *B,
                          float *C,
                          int M,
                          int N,
                          int K,
                          float alpha,
                          float beta) {
    // base_x/base_y 是当前 block 在输出矩阵 C 中负责区域的左上角。
    const int base_x = blockIdx.x * blockDim.x * kTm;
    const int base_y = blockIdx.y * blockDim.y * kTn;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // c 数组保存在寄存器中，用来累计当前线程负责的一批输出。
    float c[kTm * kTn] = {};

    // shared memory 缓存当前要计算的 A/B 子块。
    __shared__ float subA[kBlockM * kBlockK];
    __shared__ float subB[kBlockN * kBlockK];

    // regA / regB 是临时寄存器向量。
    // 用 float4 一次读取 4 个 float，可以减少标量读写指令。
    float4 regB[kTm / 4];
    float4 regA[kTm / 4];

    // baseA/baseB 指向当前 block 所需 A/B 子块在 global memory 中的起点。
    const float *baseA = A + base_y * K;
    const float *baseB = B + base_x;

    // ldb8 = N * 8，用于每轮沿 K 维推进时快速更新 B 指针。
    const int ldb8 = N << 3;

    // 下面这些索引控制“每个线程搬哪些数据”。
    const int rowA = tid >> 1;
    const int rowB = tid >> 5;
    const int colA = (tid & 1) << 2;
    const int colB = (tid << 2) & 127;

    // warp_id 表示当前线程属于哪个 warp。
    // 一个 warp 固定是 32 个线程，是 NVIDIA GPU 的基本调度单位。
    const int warp_id = tid >> 5;
    const int warp_lane = tid & 31;

    // 这两个量决定当前线程最终负责 C 子块中的哪一片区域。
    // 这部分映射逻辑不直观，但本质上是在做更细粒度的线程分工。
    const int rowC = ((warp_id >> 1 << 2) + (warp_lane & 3)) << 3;
    const int colC = (((warp_id & 1) << 3) + (warp_lane >> 2)) << 3;

    float *baseC = C + (base_y + rowC) * N + base_x + colC;

    // 和 gemm_1 一样，沿 K 维分块推进。
    for (int i = 0; i < K; i += kBlockK) {
        // 从 global memory 向寄存器做向量化读取。
        regA[0] = *reinterpret_cast<const float4 *>(baseA + rowA * K + colA);
        regB[0] = *reinterpret_cast<const float4 *>(baseB + rowB * N + colB);

        // B 这边可以直接按 float4 写到 shared memory。
        *reinterpret_cast<float4 *>(&subB[tid * 4]) = regB[0];

        // A 这边写入 shared memory 时做了一个转置式布局，
        // 目的是让后续访问模式更合适。
        subA[rowA + colA * kBlockM] = regA[0].x;
        subA[rowA + (colA + 1) * kBlockM] = regA[0].y;
        subA[rowA + (colA + 2) * kBlockM] = regA[0].z;
        subA[rowA + (colA + 3) * kBlockM] = regA[0].w;

        // 更新 global memory 指针，准备下一轮 K 子块。
        baseA += kBlockK;
        baseB += ldb8;

        __syncthreads();

#pragma unroll
        // 当前 K 子块中的计算阶段。
        for (int kk = 0; kk < kBlockK; ++kk) {
            // 从 shared memory 中把本轮需要的数据读入寄存器。
            regB[0] = *reinterpret_cast<float4 *>(&subB[colC + kBlockN * kk]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[colC + 4 + kBlockN * kk]);
            regA[0] = *reinterpret_cast<float4 *>(&subA[rowC + kk * kBlockM]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[(rowC + 4) + kk * kBlockM]);

#pragma unroll
            for (int cpi = 0; cpi < kTm / 4; ++cpi) {
#pragma unroll
                for (int cpj = 0; cpj < kTn / 4; ++cpj) {
                    // 下面这一大段展开后的乘加，本质上仍然是在做小矩阵乘法。
                    // 只是它把很多循环手工展开到寄存器级别，
                    // 以换取更高的指令级并行和更少的地址计算开销。
                    c[cpi * 4 * kTm + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * kTm + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * kTm + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * kTm + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * kTm + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * kTm + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * kTm + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * kTm + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * kTm + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * kTm + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * kTm + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * kTm + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * kTm + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * kTm + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * kTm + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * kTm + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    // 把寄存器结果写回 global memory。
    // 这里保留了 alpha/beta 接口形式，看起来更像 BLAS 的 GEMM：
    // C = alpha * A * B + beta * C
    // 当前调用里传的是 alpha=1, beta=0，也就是直接覆盖输出。
    for (int i = 0; i < kTm; ++i) {
#pragma unroll
        for (int j = 0; j < kTn; j += 4) {
            *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * N + j]);
            regA[0].x = regA[0].x * beta + alpha * c[i * kTm + j];
            regA[0].y = regA[0].y * beta + alpha * c[i * kTm + j + 1];
            regA[0].z = regA[0].z * beta + alpha * c[i * kTm + j + 2];
            regA[0].w = regA[0].w * beta + alpha * c[i * kTm + j + 3];
            *reinterpret_cast<float4 *>(&baseC[i * N + j]) = *reinterpret_cast<float4 *>(&regA[0]);
        }
    }
}

void LaunchGemm2(int m, int n, int k, const float *a, const float *b, float *c) {
    dim3 threads_per_block(kBlockSize, kBlockSize);
    dim3 num_blocks((n + kBlockN - 1) / kBlockN, (m + kBlockM - 1) / kBlockM);
    matrixMul<<<num_blocks, threads_per_block>>>(a, b, c, m, n, k, 1.0f, 0.0f);
}

// 这版也要求尺寸能被 block tile 整除。
bool ValidateGemm2(const gemm::Options &opt) {
    return gemm::ValidateDivisibility(opt, kBlockM, kBlockN, kBlockK);
}

}  // namespace

namespace gemm {

KernelSpec GetGemm2Spec() {
    return KernelSpec{"gemm_2", LaunchGemm2, ValidateGemm2};
}

}  // namespace gemm
