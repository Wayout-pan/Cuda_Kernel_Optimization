#pragma once

#include "gemm/benchmark_utils.h"

namespace gemm {

using LaunchFn = void (*)(int m, int n, int k, const float *a, const float *b, float *c);
using ValidateFn = bool (*)(const Options &opt);

struct KernelSpec {
    const char *name;
    LaunchFn launch;
    ValidateFn validate;
};

const KernelSpec *FindKernel(const char *name);
void PrintKernelList();

KernelSpec GetGemm0Spec();
KernelSpec GetGemm1Spec();
KernelSpec GetGemm2Spec();

}  // namespace gemm
