#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "gemm/benchmark_utils.h"
#include "gemm/kernels.h"

namespace {

struct RunnerOptions {
    gemm::Options common;
    std::string kernel = "gemm_0";
    bool list_kernels = false;
};

void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s --kernel NAME [--m M] [--n N] [--k K] [--warmup W] [--iters I] [--check] [--list-kernels]\n",
        prog);
}

bool ParseArgs(int argc, char **argv, RunnerOptions &opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--kernel") {
            if (i + 1 >= argc) {
                return false;
            }
            opt.kernel = argv[++i];
        } else if (arg == "--list-kernels") {
            opt.list_kernels = true;
        } else if (arg == "--m") {
            if (!gemm::ParseIntArg(argc, argv, i, opt.common.m)) return false;
        } else if (arg == "--n") {
            if (!gemm::ParseIntArg(argc, argv, i, opt.common.n)) return false;
        } else if (arg == "--k") {
            if (!gemm::ParseIntArg(argc, argv, i, opt.common.k)) return false;
        } else if (arg == "--warmup") {
            if (!gemm::ParseIntArg(argc, argv, i, opt.common.warmup)) return false;
        } else if (arg == "--iters") {
            if (!gemm::ParseIntArg(argc, argv, i, opt.common.iters)) return false;
        } else if (arg == "--check") {
            opt.common.check = true;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    RunnerOptions opt;
    if (!ParseArgs(argc, argv, opt)) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (opt.list_kernels) {
        gemm::PrintKernelList();
        return EXIT_SUCCESS;
    }

    const gemm::KernelSpec *kernel = gemm::FindKernel(opt.kernel.c_str());
    if (kernel == nullptr) {
        std::fprintf(stderr, "Unknown kernel: %s\n", opt.kernel.c_str());
        gemm::PrintKernelList();
        return EXIT_FAILURE;
    }
    if (!kernel->validate(opt.common)) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const size_t size_a = static_cast<size_t>(opt.common.m) * opt.common.k;
    const size_t size_b = static_cast<size_t>(opt.common.k) * opt.common.n;
    const size_t size_c = static_cast<size_t>(opt.common.m) * opt.common.n;

    std::vector<float> h_a(size_a);
    std::vector<float> h_b(size_b);
    std::vector<float> h_c(size_c, 0.0f);

    std::mt19937 rng(42);
    gemm::InitRandom(h_a, rng);
    gemm::InitRandom(h_b, rng);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, size_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, size_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, size_c * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < opt.common.warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_c, 0, size_c * sizeof(float)));
        kernel->launch(opt.common.m, opt.common.n, opt.common.k, d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < opt.common.iters; ++i) {
        CUDA_CHECK(cudaMemset(d_c, 0, size_c * sizeof(float)));
        kernel->launch(opt.common.m, opt.common.n, opt.common.k, d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(opt.common.iters);
    const double tflops = gemm::ComputeTflops(opt.common.m, opt.common.n, opt.common.k, avg_ms);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf(
        "RESULT kernel=%s M=%d N=%d K=%d warmup=%d iters=%d avg_ms=%.4f tflops=%.4f checksum=%.10e\n",
        kernel->name,
        opt.common.m,
        opt.common.n,
        opt.common.k,
        opt.common.warmup,
        opt.common.iters,
        avg_ms,
        tflops,
        gemm::Checksum(h_c));

    if (opt.common.check) {
        const long long ops = 1LL * opt.common.m * opt.common.n * opt.common.k;
        if (ops > 200000000LL) {
            std::printf("CHECK kernel=%s skipped=1 reason=matrix_too_large_for_cpu_reference\n", kernel->name);
        } else {
            std::vector<float> h_ref(size_c, 0.0f);
            gemm::CpuGemm(h_a, h_b, h_ref, opt.common.m, opt.common.n, opt.common.k);
            std::printf("CHECK kernel=%s skipped=0 max_abs_diff=%.8e\n",
                        kernel->name,
                        gemm::MaxAbsDiff(h_c, h_ref));
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return EXIT_SUCCESS;
}
