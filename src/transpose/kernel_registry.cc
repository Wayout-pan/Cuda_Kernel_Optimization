#include "transpose/kernels.h"

#include <cstdio>
#include <string>

namespace transpose {
namespace {

// 当前 transpose 模块的 kernel 注册表。
// 新增 transpose_X.cu 后，把 GetTransposeXSpec() 加到这里即可。
KernelSpec kKernels[] = {
    GetTranspose0Spec(),
    GetTranspose1Spec(),
    GetTranspose2Spec(),
};

constexpr int kKernelCount = sizeof(kKernels) / sizeof(kKernels[0]);

}  // namespace

// 按名字查找 kernel。
const KernelSpec *FindKernel(const char *name) {
    for (int i = 0; i < kKernelCount; ++i) {
        if (std::string(kKernels[i].name) == name) {
            return &kKernels[i];
        }
    }
    return nullptr;
}

// 打印所有已注册 kernel。
void PrintKernelList() {
    for (int i = 0; i < kKernelCount; ++i) {
        std::printf("%s\n", kKernels[i].name);
    }
}

}  // namespace transpose
