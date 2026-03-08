#include "gemm/kernels.h"

#include <cstdio>
#include <string>

namespace gemm {
namespace {

// 这里是整个工程的 kernel 注册表。
// 新增一个 kernel 的最小改动通常是两步：
// 1. 新建一个 gemm_X.cu，并提供 GetGemmXSpec()
// 2. 把它加到下面这个数组里
KernelSpec kKernels[] = {
    GetGemm0Spec(),
    GetGemm1Spec(),
    GetGemm2Spec(),
};

constexpr int kKernelCount = sizeof(kKernels) / sizeof(kKernels[0]);

}  // namespace

// 按名字查找 kernel。
// 这里是线性扫描，因为当前 kernel 数量很少，简单直接就够了。
const KernelSpec *FindKernel(const char *name) {
    for (int i = 0; i < kKernelCount; ++i) {
        if (std::string(kKernels[i].name) == name) {
            return &kKernels[i];
        }
    }
    return nullptr;
}

// 把所有 kernel 名字逐行打印出来。
// benchmark 脚本会调用这个接口自动发现当前有哪些 kernel 可跑。
void PrintKernelList() {
    for (int i = 0; i < kKernelCount; ++i) {
        std::printf("%s\n", kKernels[i].name);
    }
}

}  // namespace gemm
