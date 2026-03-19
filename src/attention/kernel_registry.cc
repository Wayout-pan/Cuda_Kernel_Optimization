#include "attention/kernels.h"

#include <cstdio>
#include <string>

namespace attention {
namespace {

// 当前 attention 模块的 kernel 注册表。
// 新增 attention_X.cu 后，把 GetAttentionXSpec() 接进来即可。
KernelSpec kKernels[] = {
    GetAttention0Spec(),
    GetAttention1Spec(),
    GetAttention2Spec(),
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

}  // namespace attention
