#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "transpose/benchmark_utils.h"
#include "transpose/kernels.h"

namespace {

// RunnerOptions 是 transpose runner 自己的参数结构。
struct RunnerOptions {
    transpose::Options common;
    std::string kernel = "transpose_0";
    bool list_kernels = false;
};

// 打印命令行帮助。
void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s --kernel NAME [--m M] [--n N] [--warmup W] [--iters I] [--check] [--list-kernels]\n",
        prog);
}

// 解析 transpose runner 的命令行参数。
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
            if (!transpose::ParseIntArg(argc, argv, i, opt.common.m)) return false;
        } else if (arg == "--n") {
            if (!transpose::ParseIntArg(argc, argv, i, opt.common.n)) return false;
        } else if (arg == "--warmup") {
            if (!transpose::ParseIntArg(argc, argv, i, opt.common.warmup)) return false;
        } else if (arg == "--iters") {
            if (!transpose::ParseIntArg(argc, argv, i, opt.common.iters)) return false;
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

    // 如果用户只想看 kernel 列表，打印后直接退出。
    if (opt.list_kernels) {
        transpose::PrintKernelList();
        return EXIT_SUCCESS;
    }

    // 从注册表中找到对应的 transpose kernel。
    const transpose::KernelSpec *kernel = transpose::FindKernel(opt.kernel.c_str());
    if (kernel == nullptr) {
        std::fprintf(stderr, "Unknown kernel: %s\n", opt.kernel.c_str());
        transpose::PrintKernelList();
        return EXIT_FAILURE;
    }

    // 检查输入是否合法。
    if (!kernel->validate(opt.common)) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // 输入矩阵和输出矩阵元素总数相同，都是 m * n。
    const size_t element_count = static_cast<size_t>(opt.common.m) * opt.common.n;
    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count, 0.0f);

    // 固定随机种子，便于复现实验。
    std::mt19937 rng(42);
    transpose::InitRandom(h_input, rng);

    // 分配 GPU 输入/输出缓冲区。
    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, element_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, element_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        d_input, h_input.data(), element_count * sizeof(float), cudaMemcpyHostToDevice));

    // 预热阶段：
    // 1. 清空输出矩阵
    // 2. 启动 kernel
    // 3. 最后同步一次
    for (int i = 0; i < opt.common.warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, element_count * sizeof(float)));
        kernel->launch(opt.common.m, opt.common.n, d_input, d_output);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 用 CUDA event 测量 GPU 时间。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < opt.common.iters; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, element_count * sizeof(float)));
        kernel->launch(opt.common.m, opt.common.n, d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(opt.common.iters);
    const double gbps = transpose::ComputeBandwidthGbps(element_count, avg_ms);

    // 拷回输出，便于打印 checksum 和做 CPU 校验。
    CUDA_CHECK(cudaMemcpy(
        h_output.data(), d_output, element_count * sizeof(float), cudaMemcpyDeviceToHost));
    // transpose 更偏 memory-bound，因此用 GB/s 更合适。
    std::printf(
        "RESULT kernel=%s M=%d N=%d warmup=%d iters=%d avg_ms=%.4f gbps=%.4f checksum=%.10e\n",
        kernel->name,
        opt.common.m,
        opt.common.n,
        opt.common.warmup,
        opt.common.iters,
        avg_ms,
        gbps,
        transpose::Checksum(h_output));

    // 可选的 CPU 参考校验。
    if (opt.common.check) {
        std::vector<float> h_ref(element_count, 0.0f);
        transpose::CpuTranspose(h_input, h_ref, opt.common.m, opt.common.n);
        std::printf("CHECK kernel=%s skipped=0 max_abs_diff=%.8e\n",
                    kernel->name,
                    transpose::MaxAbsDiff(h_output, h_ref));
    }

    // 释放资源。
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return EXIT_SUCCESS;
}
