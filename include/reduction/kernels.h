#pragma once

#include "reduction/benchmark_utils.h"

namespace reduction {

// LaunchFn 是 reduction kernel 的统一启动接口。
// 参数约定为：
// - n: 输入长度
// - input: device 端输入向量
// - output: device 端单元素输出地址
using LaunchFn = void (*)(int n, const float *input, float *output);

// ValidateFn 用来描述某个 kernel 能接受哪些输入。
// 当前 reduction_0 / reduction_1 没有额外 shape 约束，
// 但保留这层接口后，后续扩展更复杂的 kernel 会更方便。
using ValidateFn = bool (*)(const Options &opt);

// KernelSpec 是 reduction kernel 的“说明书”。
struct KernelSpec {
    const char *name;
    LaunchFn launch;
    ValidateFn validate;
};

// 按名字查找 kernel。
const KernelSpec *FindKernel(const char *name);

// 打印所有已注册 kernel 的名字。
void PrintKernelList();

// 每个实现文件都会提供一个 GetReductionXSpec()。
KernelSpec GetReduction0Spec();
KernelSpec GetReduction1Spec();

}  // namespace reduction
