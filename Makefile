NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
CXXFLAGS ?= -O3 -std=c++17
INCLUDES := -Iinclude

BUILD_DIR := build
RUNNER := $(BUILD_DIR)/gemm_runner
KERNEL_SRCS := $(sort $(wildcard src/gemm/gemm_[0-9]*.cu))
RUNNER_SRCS := src/gemm/gemm_runner.cu src/gemm/kernel_registry.cc $(KERNEL_SRCS)

.PHONY: all bench clean list

all: $(RUNNER)

list: $(RUNNER)
	@$(RUNNER) --list-kernels

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(RUNNER): $(RUNNER_SRCS) include/gemm/benchmark_utils.h include/gemm/kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) $(RUNNER_SRCS) -o $@

bench: all
	bash scripts/bench_gemm.sh

clean:
	rm -rf $(BUILD_DIR) results
