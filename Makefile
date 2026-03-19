CUDA_HOME ?= /usr/local/cuda-13
NVCC ?= $(CUDA_HOME)/bin/nvcc
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
TRANSPOSE_RUNNER := $(BUILD_DIR)/transpose_runner
TRANSPOSE_KERNEL_SRCS := $(sort $(wildcard src/transpose/transpose_[0-9]*.cu))
TRANSPOSE_RUNNER_SRCS := src/transpose/transpose_runner.cu src/transpose/kernel_registry.cc $(TRANSPOSE_KERNEL_SRCS)
TRANSPOSE_OBJS := $(BUILD_DIR)/src/transpose/transpose_runner.o \
	$(BUILD_DIR)/src/transpose/kernel_registry.o \
	$(patsubst src/transpose/%.cu,$(BUILD_DIR)/src/transpose/%.o,$(TRANSPOSE_KERNEL_SRCS))
ATTENTION_RUNNER := $(BUILD_DIR)/attention_runner
ATTENTION_KERNEL_SRCS := $(sort $(wildcard src/attention/attention_[0-9]*.cu))
ATTENTION_RUNNER_SRCS := src/attention/attention_runner.cu src/attention/kernel_registry.cc $(ATTENTION_KERNEL_SRCS)
ATTENTION_OBJS := $(BUILD_DIR)/src/attention/attention_runner.o \
	$(BUILD_DIR)/src/attention/kernel_registry.o \
	$(patsubst src/attention/%.cu,$(BUILD_DIR)/src/attention/%.o,$(ATTENTION_KERNEL_SRCS))

.PHONY: all gemm reduction transpose attention list list-gemm list-reduction list-transpose list-attention bench bench-gemm bench-reduction bench-transpose bench-attention plot clean

all: gemm reduction transpose attention

gemm: $(GEMM_RUNNER)

reduction: $(REDUCTION_RUNNER)

transpose: $(TRANSPOSE_RUNNER)

attention: $(ATTENTION_RUNNER)

list: gemm reduction transpose attention
	@echo "[gemm]"
	@$(GEMM_RUNNER) --list-kernels
	@echo
	@echo "[reduction]"
	@$(REDUCTION_RUNNER) --list-kernels
	@echo
	@echo "[transpose]"
	@$(TRANSPOSE_RUNNER) --list-kernels
	@echo
	@echo "[attention]"
	@$(ATTENTION_RUNNER) --list-kernels

list-gemm: $(GEMM_RUNNER)
	@$(GEMM_RUNNER) --list-kernels

list-reduction: $(REDUCTION_RUNNER)
	@$(REDUCTION_RUNNER) --list-kernels

list-transpose: $(TRANSPOSE_RUNNER)
	@$(TRANSPOSE_RUNNER) --list-kernels

list-attention: $(ATTENTION_RUNNER)
	@$(ATTENTION_RUNNER) --list-kernels

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(GEMM_RUNNER): $(GEMM_RUNNER_SRCS) include/gemm/benchmark_utils.h include/gemm/kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) $(GEMM_RUNNER_SRCS) -o $@

$(REDUCTION_RUNNER): $(REDUCTION_RUNNER_SRCS) include/reduction/benchmark_utils.h include/reduction/kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) $(REDUCTION_RUNNER_SRCS) -o $@

$(BUILD_DIR)/src/transpose/%.o: src/transpose/%.cu include/transpose/benchmark_utils.h include/transpose/kernels.h | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/transpose/kernel_registry.o: src/transpose/kernel_registry.cc include/transpose/kernels.h include/transpose/benchmark_utils.h | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TRANSPOSE_RUNNER): $(TRANSPOSE_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/src/attention/%.o: src/attention/%.cu include/attention/benchmark_utils.h include/attention/kernels.h | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/src/attention/kernel_registry.o: src/attention/kernel_registry.cc include/attention/kernels.h include/attention/benchmark_utils.h | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(ATTENTION_RUNNER): $(ATTENTION_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $^ -o $@

bench: bench-gemm bench-reduction bench-transpose bench-attention

bench-gemm: gemm
	bash scripts/bench_gemm.sh

bench-reduction: reduction
	bash scripts/bench_reduction.sh

bench-transpose: transpose
	bash scripts/bench_transpose.sh

bench-attention: attention
	bash scripts/bench_attention.sh

plot:
	python3 scripts/plot_benchmarks.py

clean:
	rm -rf $(BUILD_DIR) results
