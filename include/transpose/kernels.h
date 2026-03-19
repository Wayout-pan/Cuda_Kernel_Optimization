#pragma once

#include "transpose/benchmark_utils.h"

namespace transpose {

// transpose kernel 的统一 launch 接口。
// 参数约定为：
// - m, n: 输入矩阵 shape = [m x n]
// - input: device 端输入矩阵
// - output: device 端输出矩阵，shape = [n x m]
using LaunchFn = void (*)(int m, int n, const float *input, float *output);

// ValidateFn 描述每个 kernel 支持哪些输入。
using ValidateFn = bool (*)(const Options &opt);

// KernelSpec 是 transpose kernel 的说明书。
struct KernelSpec {
    const char *name;
    LaunchFn launch;
    ValidateFn validate;
};

// 按名字查找 kernel。
const KernelSpec *FindKernel(const char *name);

// 打印当前所有已注册 kernel。
void PrintKernelList();

// 每个 transpose_X.cu 都会导出自己的 spec。
KernelSpec GetTranspose0Spec();
KernelSpec GetTranspose1Spec();
KernelSpec GetTranspose2Spec();

}  // namespace transpose
