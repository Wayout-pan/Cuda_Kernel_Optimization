#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                                 \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

namespace {

constexpr int kBlockSize = 16;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kTm = kBlockM / kBlockSize;
constexpr int kTn = kBlockN / kBlockSize;

struct Options {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 10;
    int iters = 50;
    bool check = false;
};

void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s [--m M] [--n N] [--k K] [--warmup W] [--iters I] [--check]\\n",
        prog);
}

bool ParseIntArg(int argc, char **argv, int &idx, int &dst) {
    if (idx + 1 >= argc) {
        return false;
    }
    dst = std::atoi(argv[idx + 1]);
    idx++;
    return true;
}

bool ParseArgs(int argc, char **argv, Options &opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--m") {
            if (!ParseIntArg(argc, argv, i, opt.m)) return false;
        } else if (arg == "--n") {
            if (!ParseIntArg(argc, argv, i, opt.n)) return false;
        } else if (arg == "--k") {
            if (!ParseIntArg(argc, argv, i, opt.k)) return false;
        } else if (arg == "--warmup") {
            if (!ParseIntArg(argc, argv, i, opt.warmup)) return false;
        } else if (arg == "--iters") {
            if (!ParseIntArg(argc, argv, i, opt.iters)) return false;
        } else if (arg == "--check") {
            opt.check = true;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            return false;
        }
    }
    return true;
}

bool ValidateShape(const Options &opt) {
    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.warmup < 0 || opt.iters <= 0) {
        return false;
    }
    if (opt.m % kBlockM != 0 || opt.n % kBlockN != 0 || opt.k % kBlockK != 0) {
        std::fprintf(stderr,
                     "Shape constraints not satisfied: require M %% %d == 0, N %% %d == 0, K %% %d == 0.\\n",
                     kBlockM,
                     kBlockN,
                     kBlockK);
        return false;
    }
    return true;
}

float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

void CpuGemm(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c,
            int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += a[OFFSET(row, kk, k)] * b[OFFSET(kk, col, n)];
            }
            c[OFFSET(row, col, n)] = acc;
        }
    }
}

}  // namespace

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

    int rowA = tid >> 1;
    int rowB = tid >> 5;
    int colA = (tid & 1) << 2;
    int colB = (tid << 2) & 127;

    int warp_id = tid >> 5;
    int warp_lane = tid & 31;
    int rowC = ((warp_id >> 1 << 2) + (warp_lane & 3)) << 3;
    int colC = (((warp_id & 1) << 3) + (warp_lane >> 2)) << 3;

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
        for (int kk = 0; kk < kBlockK; kk++) {
            regB[0] = *reinterpret_cast<float4 *>(&subB[colC + kBlockN * kk]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[colC + 4 + kBlockN * kk]);
            regA[0] = *reinterpret_cast<float4 *>(&subA[rowC + kk * kBlockM]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[(rowC + 4) + kk * kBlockM]);

#pragma unroll
            for (int cpi = 0; cpi < kTm / 4; cpi++) {
#pragma unroll
                for (int cpj = 0; cpj < kTn / 4; cpj++) {
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
    for (int i = 0; i < kTm; i++) {
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

void sgemm_v2(int M, int N, int K, const float *a, const float *b, float *c, float alpha = 1.0f, float beta = 0.0f) {
    dim3 threads_per_block(kBlockSize, kBlockSize);
    dim3 num_blocks((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
    matrixMul<<<num_blocks, threads_per_block>>>(a, b, c, M, N, K, alpha, beta);
}

int main(int argc, char **argv) {
    Options opt;
    if (!ParseArgs(argc, argv, opt) || !ValidateShape(opt)) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const size_t size_a = static_cast<size_t>(opt.m) * opt.k;
    const size_t size_b = static_cast<size_t>(opt.k) * opt.n;
    const size_t size_c = static_cast<size_t>(opt.m) * opt.n;

    std::vector<float> h_a(size_a);
    std::vector<float> h_b(size_b);
    std::vector<float> h_c(size_c, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : h_a) v = dist(rng);
    for (float &v : h_b) v = dist(rng);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, size_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, size_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, size_c * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, size_c * sizeof(float)));

    for (int i = 0; i < opt.warmup; ++i) {
        sgemm_v2(opt.m, opt.n, opt.k, d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < opt.iters; ++i) {
        sgemm_v2(opt.m, opt.n, opt.k, d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(opt.iters);
    const double tflops = (2.0 * static_cast<double>(opt.m) * opt.n * opt.k) /
                          (static_cast<double>(avg_ms) * 1e9);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_c) checksum += v;

    std::printf(
        "RESULT kernel=gemm_2 M=%d N=%d K=%d warmup=%d iters=%d avg_ms=%.4f tflops=%.4f checksum=%.10e\\n",
        opt.m,
        opt.n,
        opt.k,
        opt.warmup,
        opt.iters,
        avg_ms,
        tflops,
        checksum);

    if (opt.check) {
        const long long ops = 1LL * opt.m * opt.n * opt.k;
        if (ops > 200000000LL) {
            std::printf("CHECK kernel=gemm_2 skipped=1 reason=matrix_too_large_for_cpu_reference\\n");
        } else {
            std::vector<float> h_ref(size_c, 0.0f);
            CpuGemm(h_a, h_b, h_ref, opt.m, opt.n, opt.k);
            const float max_diff = MaxAbsDiff(h_c, h_ref);
            std::printf("CHECK kernel=gemm_2 skipped=0 max_abs_diff=%.8e\\n", max_diff);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}
