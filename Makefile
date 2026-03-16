NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
CXXFLAGS ?= -O3 -std=c++17
INCLUDES := -Iinclude

BUILD_DIR := build
GEMM_RUNNER := $(BUILD_DIR)/gemm_runner
GEMM_KERNEL_SRCS := $(sort $(wildcard src/gemm/gemm_[0-9]*.cu))
GEMM_RUNNER_SRCS := src/gemm/gemm_runner.cu src/gemm/kernel_registry.cc $(GEMM_KERNEL_SRCS)
REDUCTION_RUNNER := $(BUILD_DIR)/reduction_runner
REDUCTION_KERNEL_SRCS := $(sort $(wildcard src/reduction/reduction_[0-9]*.cu))
REDUCTION_RUNNER_SRCS := src/reduction/reduction_runner.cu src/reduction/kernel_registry.cc $(REDUCTION_KERNEL_SRCS)

.PHONY: all gemm reduction list list-gemm list-reduction bench bench-gemm bench-reduction clean

all: gemm reduction

gemm: $(GEMM_RUNNER)

reduction: $(REDUCTION_RUNNER)

list: gemm reduction
	@echo "[gemm]"
	@$(GEMM_RUNNER) --list-kernels
	@echo
	@echo "[reduction]"
	@$(REDUCTION_RUNNER) --list-kernels

list-gemm: $(GEMM_RUNNER)
	@$(GEMM_RUNNER) --list-kernels

list-reduction: $(REDUCTION_RUNNER)
	@$(REDUCTION_RUNNER) --list-kernels

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(GEMM_RUNNER): $(GEMM_RUNNER_SRCS) include/gemm/benchmark_utils.h include/gemm/kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) $(GEMM_RUNNER_SRCS) -o $@

$(REDUCTION_RUNNER): $(REDUCTION_RUNNER_SRCS) include/reduction/benchmark_utils.h include/reduction/kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) $(REDUCTION_RUNNER_SRCS) -o $@

bench: bench-gemm bench-reduction

bench-gemm: gemm
	bash scripts/bench_gemm.sh

bench-reduction: reduction
	bash scripts/bench_reduction.sh

clean:
	rm -rf $(BUILD_DIR) results
