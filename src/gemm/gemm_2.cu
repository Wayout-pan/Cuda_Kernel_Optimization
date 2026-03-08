#include <cuda_runtime.h>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

namespace {

constexpr int kBlockSize = 16;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
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
    const int base_x = blockIdx.x * blockDim.x * kTm;
    const int base_y = blockIdx.y * blockDim.y * kTn;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float c[kTm * kTn] = {};

    __shared__ float subA[kBlockM * kBlockK];
    __shared__ float subB[kBlockN * kBlockK];

    float4 regB[kTm / 4];
    float4 regA[kTm / 4];

    const float *baseA = A + base_y * K;
    const float *baseB = B + base_x;
    const int ldb8 = N << 3;

    const int rowA = tid >> 1;
    const int rowB = tid >> 5;
    const int colA = (tid & 1) << 2;
    const int colB = (tid << 2) & 127;

    const int warp_id = tid >> 5;
    const int warp_lane = tid & 31;
    const int rowC = ((warp_id >> 1 << 2) + (warp_lane & 3)) << 3;
    const int colC = (((warp_id & 1) << 3) + (warp_lane >> 2)) << 3;

    float *baseC = C + (base_y + rowC) * N + base_x + colC;

    for (int i = 0; i < K; i += kBlockK) {
        regA[0] = *reinterpret_cast<const float4 *>(baseA + rowA * K + colA);
        regB[0] = *reinterpret_cast<const float4 *>(baseB + rowB * N + colB);

        *reinterpret_cast<float4 *>(&subB[tid * 4]) = regB[0];

        subA[rowA + colA * kBlockM] = regA[0].x;
        subA[rowA + (colA + 1) * kBlockM] = regA[0].y;
        subA[rowA + (colA + 2) * kBlockM] = regA[0].z;
        subA[rowA + (colA + 3) * kBlockM] = regA[0].w;

        baseA += kBlockK;
        baseB += ldb8;

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            regB[0] = *reinterpret_cast<float4 *>(&subB[colC + kBlockN * kk]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[colC + 4 + kBlockN * kk]);
            regA[0] = *reinterpret_cast<float4 *>(&subA[rowC + kk * kBlockM]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[(rowC + 4) + kk * kBlockM]);

#pragma unroll
            for (int cpi = 0; cpi < kTm / 4; ++cpi) {
#pragma unroll
                for (int cpj = 0; cpj < kTn / 4; ++cpj) {
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

bool ValidateGemm2(const gemm::Options &opt) {
    return gemm::ValidateDivisibility(opt, kBlockM, kBlockN, kBlockK);
}

}  // namespace

namespace gemm {

KernelSpec GetGemm2Spec() {
    return KernelSpec{"gemm_2", LaunchGemm2, ValidateGemm2};
}

}  // namespace gemm
