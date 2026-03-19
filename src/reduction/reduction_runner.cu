#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "reduction/benchmark_utils.h"
#include "reduction/kernels.h"

namespace {

// RunnerOptions 是 runner 自己的参数结构。
// common 部分是所有 reduction kernel 都共享的公共参数。
struct RunnerOptions {
    reduction::Options common;
    std::string kernel = "reduction_0";
    bool list_kernels = false;
};

// 打印 runner 的帮助信息。
void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s --kernel NAME [--n N] [--warmup W] [--iters I] [--check] [--list-kernels]\n",
        prog);
}

// 解析 reduction runner 的命令行参数。
// 和 benchmark_utils.h 里的 ParseIntArg 配合使用，
// 这里只额外处理 --kernel 和 --list-kernels。
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
        } else if (arg == "--n") {
            if (!reduction::ParseIntArg(argc, argv, i, opt.common.n)) return false;
        } else if (arg == "--warmup") {
            if (!reduction::ParseIntArg(argc, argv, i, opt.common.warmup)) return false;
        } else if (arg == "--iters") {
            if (!reduction::ParseIntArg(argc, argv, i, opt.common.iters)) return false;
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
        reduction::PrintKernelList();
        return EXIT_SUCCESS;
    }

    // 从注册表中查找对应的 kernel。
    const reduction::KernelSpec *kernel = reduction::FindKernel(opt.kernel.c_str());
    if (kernel == nullptr) {
        std::fprintf(stderr, "Unknown kernel: %s\n", opt.kernel.c_str());
        reduction::PrintKernelList();
        return EXIT_FAILURE;
    }

    // 调用当前 kernel 自己的输入约束检查。
    if (!kernel->validate(opt.common)) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // reduction 的输入是长度为 N 的向量。
    const size_t size_input = static_cast<size_t>(opt.common.n);
    std::vector<float> h_input(size_input);

    // 固定随机种子，保证实验可复现。
    std::mt19937 rng(42);
    reduction::InitRandom(h_input, rng);

    // d_output 只有一个 float，用来接收最终 sum。
    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input * sizeof(float), cudaMemcpyHostToDevice));

    // 预热阶段：
    // 1. 清零输出
    // 2. 启动 kernel
    // 3. 最后同步一次，确保预热全部结束
    for (int i = 0; i < opt.common.warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        kernel->launch(opt.common.n, d_input, d_output);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 用 CUDA event 计时，测的是 GPU 侧执行时间。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < opt.common.iters; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        kernel->launch(opt.common.n, d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(opt.common.iters);

    // 拷回最终 sum，便于打印和正确性校验。
    float h_output = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    const double gbps = reduction::ComputeBandwidthGbps(size_input * sizeof(float), avg_ms);

    // reduction 更适合输出带宽指标，因为它更偏 memory-bound。
    std::printf(
        "RESULT kernel=%s N=%d warmup=%d iters=%d avg_ms=%.4f gbps=%.4f value=%.10e\n",
        kernel->name,
        opt.common.n,
        opt.common.warmup,
        opt.common.iters,
        avg_ms,
        gbps,
        static_cast<double>(h_output));

    // 可选的 CPU 参考校验。
    if (opt.common.check) {
        const double ref = reduction::CpuReduce(h_input);
        const double actual = static_cast<double>(h_output);
        std::printf("CHECK kernel=%s skipped=0 abs_diff=%.8e rel_diff=%.8e\n",
                    kernel->name,
                    reduction::AbsDiff(actual, ref),
                    reduction::RelDiff(actual, ref));
    }

    // 释放资源是标准收尾动作。
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return EXIT_SUCCESS;
}
