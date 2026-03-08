#include "gemm/kernels.h"

#include <cstdio>

namespace gemm {
namespace {

KernelSpec kKernels[] = {
    GetGemm0Spec(),
    GetGemm1Spec(),
    GetGemm2Spec(),
};

constexpr int kKernelCount = sizeof(kKernels) / sizeof(kKernels[0]);

}  // namespace

const KernelSpec *FindKernel(const char *name) {
    for (int i = 0; i < kKernelCount; ++i) {
        if (std::string(kKernels[i].name) == name) {
            return &kKernels[i];
        }
    }
    return nullptr;
}

void PrintKernelList() {
    for (int i = 0; i < kKernelCount; ++i) {
        std::printf("%s\n", kKernels[i].name);
    }
}

}  // namespace gemm
