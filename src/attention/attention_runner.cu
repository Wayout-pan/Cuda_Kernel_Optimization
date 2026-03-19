#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "attention/benchmark_utils.h"
#include "attention/kernels.h"

namespace {

// RunnerOptions 是 attention runner 的参数结构。
// common 部分描述 attention 的 shape 和 benchmark 参数，
// kernel 则表示当前要运行哪个实现。
struct RunnerOptions {
    attention::Options common;
    std::string kernel = "attention_0";
    bool list_kernels = false;
};

// 打印命令行帮助。
void PrintUsage(const char *prog) {
    std::printf(
        "Usage: %s --kernel NAME [--batch B] [--heads H] [--seq-len N] [--head-dim D] [--warmup W] [--iters I] [--check] [--list-kernels]\n",
        prog);
}

// 解析 attention runner 的命令行参数。
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
        } else if (arg == "--batch") {
            if (!attention::ParseIntArg(argc, argv, i, opt.common.batch)) return false;
        } else if (arg == "--heads") {
            if (!attention::ParseIntArg(argc, argv, i, opt.common.heads)) return false;
        } else if (arg == "--seq-len") {
            if (!attention::ParseIntArg(argc, argv, i, opt.common.seq_len)) return false;
        } else if (arg == "--head-dim") {
            if (!attention::ParseIntArg(argc, argv, i, opt.common.head_dim)) return false;
        } else if (arg == "--warmup") {
            if (!attention::ParseIntArg(argc, argv, i, opt.common.warmup)) return false;
        } else if (arg == "--iters") {
            if (!attention::ParseIntArg(argc, argv, i, opt.common.iters)) return false;
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

    // 如果只想看 kernel 列表，打印后直接退出。
    if (opt.list_kernels) {
        attention::PrintKernelList();
        return EXIT_SUCCESS;
    }

    // 从注册表查找目标 kernel。
    const attention::KernelSpec *kernel = attention::FindKernel(opt.kernel.c_str());
    if (kernel == nullptr) {
        std::fprintf(stderr, "Unknown kernel: %s\n", opt.kernel.c_str());
        attention::PrintKernelList();
        return EXIT_FAILURE;
    }

    // 调用当前 kernel 自己的输入约束检查。
    if (!kernel->validate(opt.common)) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // 一个 Q/K/V/O 张量的元素总数都是 B * H * N * D。
    const int total_heads = opt.common.batch * opt.common.heads;
    const size_t tensor_elems =
        static_cast<size_t>(total_heads) * opt.common.seq_len * opt.common.head_dim;

    // host 端缓冲区：
    // h_q / h_k / h_v 是输入，h_o 是输出。
    std::vector<float> h_q(tensor_elems);
    std::vector<float> h_k(tensor_elems);
    std::vector<float> h_v(tensor_elems);
    std::vector<float> h_o(tensor_elems, 0.0f);

    // 固定随机种子，保证 benchmark 可复现。
    std::mt19937 rng(42);
    attention::InitRandom(h_q, rng);
    attention::InitRandom(h_k, rng);
    attention::InitRandom(h_v, rng);

    // 在 device 端分别申请 Q/K/V/O 缓冲区。
    float *d_q = nullptr;
    float *d_k = nullptr;
    float *d_v = nullptr;
    float *d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, tensor_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, tensor_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, tensor_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, tensor_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), tensor_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), tensor_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), tensor_elems * sizeof(float), cudaMemcpyHostToDevice));

    // 预热阶段。
    for (int i = 0; i < opt.common.warmup; ++i) {
        kernel->launch(opt.common.batch,
                       opt.common.heads,
                       opt.common.seq_len,
                       opt.common.head_dim,
                       d_q,
                       d_k,
                       d_v,
                       d_o);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 用 CUDA event 计时。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < opt.common.iters; ++i) {
        kernel->launch(opt.common.batch,
                       opt.common.heads,
                       opt.common.seq_len,
                       opt.common.head_dim,
                       d_q,
                       d_k,
                       d_v,
                       d_o);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(opt.common.iters);
    const double tflops = attention::ComputeTflops(opt.common, avg_ms);

    // 把输出从 GPU 拷回 CPU，便于打印 checksum 和做校验。
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, tensor_elems * sizeof(float), cudaMemcpyDeviceToHost));
    // attention 这里采用近似 FLOPs 换算 TFLOPS。
    std::printf(
        "RESULT kernel=%s B=%d H=%d N=%d D=%d warmup=%d iters=%d avg_ms=%.4f tflops=%.4f checksum=%.10e\n",
        kernel->name,
        opt.common.batch,
        opt.common.heads,
        opt.common.seq_len,
        opt.common.head_dim,
        opt.common.warmup,
        opt.common.iters,
        avg_ms,
        tflops,
        attention::Checksum(h_o));

    // 可选的 CPU 参考校验。
    // attention 的 CPU 参考代价很高，因此这里设置了一个阈值，太大就跳过。
    if (opt.common.check) {
        if (attention::ApproxFlops(opt.common) > 2000000000LL) {
            std::printf("CHECK kernel=%s skipped=1 reason=attention_case_too_large_for_cpu_reference\n",
                        kernel->name);
        } else {
            std::vector<float> h_ref(tensor_elems, 0.0f);
            attention::CpuAttention(h_q,
                                    h_k,
                                    h_v,
                                    h_ref,
                                    opt.common.batch,
                                    opt.common.heads,
                                    opt.common.seq_len,
                                    opt.common.head_dim);
            std::printf("CHECK kernel=%s skipped=0 max_abs_diff=%.8e\n",
                        kernel->name,
                        attention::MaxAbsDiff(h_o, h_ref));
        }
    }

    // 释放资源。
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    return EXIT_SUCCESS;
}
