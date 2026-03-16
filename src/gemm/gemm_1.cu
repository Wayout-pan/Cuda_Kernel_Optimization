#include <cuda_runtime.h>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

// 这是一个二维数组下标转一维地址的辅助宏。
// 因为 GPU 内存本质上是一维线性地址，
// 所以访问 A[row][col] 时，本质上要手动算成 row * ld + col。
// ld 可以理解为 leading dimension，也就是一行的跨度。
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 这两个宏用于把 4 个连续 float 当成一个 float4 来读写。
// 这样做的主要目的是：
// 1. 减少指令数
// 2. 更容易形成连续向量化访问
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])

namespace {

// 这里是一组 tile 参数。
// 可以把它理解成：一个 block 打算一次性计算多大的一块输出矩阵。
constexpr int kBm = 128;  // block 负责的 C 子块行数
constexpr int kBn = 128;  // block 负责的 C 子块列数
constexpr int kBk = 8;    // 每次沿 K 维推进的深度
constexpr int kTm = 8;    // 单个线程负责的 C 子块行数
constexpr int kTn = 8;    // 单个线程负责的 C 子块列数

// 这是一个典型的 shared-memory tiled GEMM。
// 相比朴素版本，它的核心优化思想是：
// 1. 一个 block 先把 A/B 的小块搬到 shared memory
// 2. block 内所有线程复用这些数据
// 3. 每个线程在寄存器里累计一个 8x8 的小结果块
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

    // tid 是当前线程在 block 内的一维编号。
    // 把二维 threadIdx 映射成一维后，更方便分配“谁负责搬哪一段数据”。
    const int tid = ty * blockDim.x + tx;

    // shared memory 是 block 内共享的片上存储。
    // 它比 global memory 快很多，但容量较小。
    // 这里分别缓存 A 的 [128 x 8] 子块和 B 的 [8 x 128] 子块。
    __shared__ float s_a[kBm][kBk];
    __shared__ float s_b[kBk][kBn];

    // r_c 放在寄存器里，保存当前线程负责的 8x8 输出子块。
    // 寄存器是线程私有的，速度最快，但数量有限。
    float r_c[kTm][kTn] = {0.0f};

    // 下面这些索引用来决定：
    // “当前线程应该从 global memory 中搬哪 4 个 float 到 shared memory”。
    //
    // A 子块大小是 128 x 8 = 1024 个 float。
    // 一个 block 有 (128/8) x (128/8) = 16 x 16 = 256 个线程。
    // 因此平均每个线程需要搬 1024 / 256 = 4 个 float。
    // 
    const int load_a_smem_m = tid >> 1; //等价于 tid / 2：2个线程搬1行数据(8个)，/2就可以得到当前线程搬第几行
    const int load_a_smem_k = (tid & 1) << 2; // 等价于tid % 2 == 1 ? 4 : 0 四个四个搬，因此k的起始要么是0要么是4，第零个线程从k=0开始搬运，第二个从k=4开始，第三个又从0开始
    const int load_b_smem_k = tid >> 5; // 等价于 tid / 32 因为一行有128个，32*4=128,因此/32得到该tid搬运第几行的
    const int load_b_smem_n = (tid & 31) << 2; // 等价于(tid % 32) * 4, tid%32表示按照4个数据为一个单位，该搬第几个单位，*4就是恢复到了原本的列(n)索引
    // 计算这批数据在 global memory 中的起始位置。
    const int load_a_gmem_m = by * kBm + load_a_smem_m;
    const int load_b_gmem_n = bx * kBn + load_b_smem_n;

    // GEMM 本质上要沿 K 维做累加。
    // 这里每次前进 kBk=8，也就是把 K 维切成很多小段。
    for (int bk = 0; bk < (k + kBk - 1) / kBk; ++bk) {
        const int load_a_gmem_k = bk * kBk + load_a_smem_k;
        const int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, k);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4_CONST(a[load_a_gmem_addr]); //将a中load_a_gmem_addr地址的连续四个数据搬到对应s_a对应的位置

        const int load_b_gmem_k = bk * kBk + load_b_smem_k;
        const int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, n);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4_CONST(b[load_b_gmem_addr]);

        // __syncthreads() 是 block 内同步。
        // 它保证所有线程都已经把 shared memory 填好，
        // 后面再开始读，避免读到未写完的数据。
        __syncthreads();

#pragma unroll
        // 下面是计算阶段。
        // 对当前这一个 K 子块中的每个 kk，
        // 当前线程都拿 shared memory 中的一行/一列数据做乘加。
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

        // 下一轮覆盖 shared memory 之前，也要先等所有线程把当前轮计算做完。
        __syncthreads();
    }

#pragma unroll
    // 所有 K 子块累加完之后，把寄存器里的结果写回 global memory。
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

// block 维度是 (16, 16)，也就是 256 个线程。
// 为什么是这样？
// 因为每个线程负责 8x8 的结果块，
// 16 x 8 = 128，因此整个 block 覆盖 128x128 的输出区域。
void LaunchGemm1(int m, int n, int k, const float *a, const float *b, float *c) {
    dim3 block(kBn / kTn, kBm / kTm);
    dim3 grid((n + kBn - 1) / kBn, (m + kBm - 1) / kBm);
    sgemm_V1<<<grid, block>>>(a, b, c, m, n, k);
}

// 这个 kernel 的实现默认要求 M/N/K 分别能被 128/128/8 整除。
// 如果不满足，就需要额外的边界处理代码。
bool ValidateGemm1(const gemm::Options &opt) {
    return gemm::ValidateDivisibility(opt, kBm, kBn, kBk);
}

}  // namespace

namespace gemm {

KernelSpec GetGemm1Spec() {
    return KernelSpec{"gemm_1", LaunchGemm1, ValidateGemm1};
}

}  // namespace gemm
