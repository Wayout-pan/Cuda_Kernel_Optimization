#include <cuda_runtime.h>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])

namespace {

constexpr int kBm = 128;
constexpr int kBn = 128;
constexpr int kBk = 8;
constexpr int kTm = 8;
constexpr int kTn = 8;

__global__ void sgemm_V1(const float *__restrict__ a,
                         const float *__restrict__ b,
                         float *__restrict__ c,
                         int m,
                         int n,
                         int k) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[kBm][kBk];
    __shared__ float s_b[kBk][kBn];

    float r_c[kTm][kTn] = {0.0f};

    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 2;
    const int load_b_smem_k = tid >> 5;
    const int load_b_smem_n = (tid & 31) << 2;

    const int load_a_gmem_m = by * kBm + load_a_smem_m;
    const int load_b_gmem_n = bx * kBn + load_b_smem_n;

    for (int bk = 0; bk < (k + kBk - 1) / kBk; ++bk) {
        const int load_a_gmem_k = bk * kBk + load_a_smem_k;
        const int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, k);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4_CONST(a[load_a_gmem_addr]);

        const int load_b_gmem_k = bk * kBk + load_b_smem_k;
        const int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, n);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4_CONST(b[load_b_gmem_addr]);

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBk; ++kk) {
#pragma unroll
            for (int mm = 0; mm < kTm; ++mm) {
#pragma unroll
                for (int nn = 0; nn < kTn; ++nn) {
                    const int comp_a_smem_m = ty * kTm + mm;
                    const int comp_b_smem_n = tx * kTn + nn;
                    r_c[mm][nn] += s_a[comp_a_smem_m][kk] * s_b[kk][comp_b_smem_n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTm; ++i) {
        const int store_c_gmem_m = by * kBm + ty * kTm + i;
#pragma unroll
        for (int j = 0; j < kTn; j += 4) {
            const int store_c_gmem_n = bx * kBn + tx * kTn + j;
            const int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, n);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

void LaunchGemm1(int m, int n, int k, const float *a, const float *b, float *c) {
    dim3 block(kBn / kTn, kBm / kTm);
    dim3 grid((n + kBn - 1) / kBn, (m + kBm - 1) / kBm);
    sgemm_V1<<<grid, block>>>(a, b, c, m, n, k);
}

bool ValidateGemm1(const gemm::Options &opt) {
    return gemm::ValidateDivisibility(opt, kBm, kBn, kBk);
}

}  // namespace

namespace gemm {

KernelSpec GetGemm1Spec() {
    return KernelSpec{"gemm_1", LaunchGemm1, ValidateGemm1};
}

}  // namespace gemm
