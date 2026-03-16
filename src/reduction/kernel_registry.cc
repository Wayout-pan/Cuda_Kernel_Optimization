#include "reduction/kernels.h"

#include <cstdio>
#include <string>

namespace reduction {
namespace {

KernelSpec kKernels[] = {
    GetReduction0Spec(),
    GetReduction1Spec(),
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

}  // namespace reduction
