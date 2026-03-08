#pragma once

// 这个头文件定义的是“所有 GEMM kernel 都必须遵守的统一接口”。
// 有了这层接口，runner 就不需要知道某个 kernel 的内部实现细节，
// 只需要拿到一个 KernelSpec，就能统一调用。

#include "gemm/benchmark_utils.h"

namespace gemm {

// LaunchFn 是一个函数指针类型。
// 它代表“如何启动一个 kernel”。
// 参数统一为：
// - m, n, k: GEMM 尺寸
// - a, b, c: 设备端(device)指针，不是主机端(host)指针
using LaunchFn = void (*)(int m, int n, int k, const float *a, const float *b, float *c);

// ValidateFn 用于检查某个 kernel 是否支持当前输入尺寸。
// 比如某些优化版本只支持 128 对齐的矩阵。
using ValidateFn = bool (*)(const Options &opt);

// KernelSpec 是 kernel 的“说明书”。
// name: 这个 kernel 的名字，供命令行和 benchmark 使用
// launch: 如何启动它
// validate: 它支持哪些输入
struct KernelSpec {
    const char *name;
    LaunchFn launch;
    ValidateFn validate;
};

// 根据名字查找 kernel。
const KernelSpec *FindKernel(const char *name);

// 打印所有已注册 kernel 的名字。
void PrintKernelList();

// 每个 .cu 文件都会提供一个对应的 GetGemmXSpec()。
// 这样 registry 可以把它们收集起来。
KernelSpec GetGemm0Spec();
KernelSpec GetGemm1Spec();
KernelSpec GetGemm2Spec();

}  // namespace gemm
