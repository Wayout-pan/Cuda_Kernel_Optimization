#pragma once

#include "attention/benchmark_utils.h"

namespace attention {

// attention kernel 的统一 launch 接口。
// 参数约定为：
// - batch, heads, seq_len, head_dim: attention shape
// - q, k, v: device 端输入张量
// - o: device 端输出张量
using LaunchFn = void (*)(int batch,
                          int heads,
                          int seq_len,
                          int head_dim,
                          const float *q,
                          const float *k,
                          const float *v,
                          float *o);

// ValidateFn 描述某个 kernel 的 shape 限制。
using ValidateFn = bool (*)(const Options &opt);

// KernelSpec 是 attention kernel 的说明书。
struct KernelSpec {
    const char *name;
    LaunchFn launch;
    ValidateFn validate;
};

// 按名字查找 kernel。
const KernelSpec *FindKernel(const char *name);

// 打印当前所有已注册 kernel。
void PrintKernelList();

// 每个 attention_X.cu 都会提供自己的 spec。
KernelSpec GetAttention0Spec();
KernelSpec GetAttention1Spec();
KernelSpec GetAttention2Spec();

}  // namespace attention
