# CUDA Kernel Optimization for GEMM

这个仓库用于做单精度 GEMM (`C = A x B`) 的 CUDA 性能优化实验。

## 结构

- `src/gemm/gemm_*.cu`: 只保留各自的 kernel 实现、launch 函数和 shape 约束。
- `src/gemm/gemm_runner.cu`: 唯一的运行入口，负责参数解析、显存管理、计时和结果校验。
- `src/gemm/kernel_registry.cc`: 注册所有可用 kernel。
- `include/gemm/benchmark_utils.h`: 通用 benchmark 工具。
- `include/gemm/kernels.h`: kernel 统一接口。
- `scripts/bench_gemm.sh`: 批量 benchmark 和 CSV 汇总。

## 使用

编译：

```bash
make all
```

查看可用 kernel：

```bash
make list
```

单独运行某个 kernel：

```bash
./build/gemm_runner --kernel gemm_1 --m 1024 --n 1024 --k 1024 --warmup 5 --iters 20 --check
```

批量 benchmark：

```bash
make bench
```

自定义 case：

```bash
CASES="512x512x512 1024x2048x1024" WARMUP=5 ITERS=20 VERIFY=1 bash scripts/bench_gemm.sh
```
