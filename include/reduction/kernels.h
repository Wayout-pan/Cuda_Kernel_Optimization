#pragma once

#include "reduction/benchmark_utils.h"

namespace reduction {

using LaunchFn = void (*)(int n, const float *input, float *output);
using ValidateFn = bool (*)(const Options &opt);

struct KernelSpec {
    const char *name;
    LaunchFn launch;
    ValidateFn validate;
};

const KernelSpec *FindKernel(const char *name);
void PrintKernelList();

KernelSpec GetReduction0Spec();
KernelSpec GetReduction1Spec();

}  // namespace reduction
