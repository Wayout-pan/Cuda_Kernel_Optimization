# CUDA Kernel Optimization for GEMM

这个仓库用于做单精度 GEMM (`C = A x B`) 的 CUDA 性能优化实验。

## 结构

- `src/gemm/gemm_*.cu`: 只保留各自的 kernel 实现、launch 函数和 shape 约束。
- `src/gemm/gemm_runner.cu`: 唯一的运行入口，负责参数解析、显存管理、计时和结果校验。
- `src/gemm/kernel_registry.cc`: 注册所有可用 kernel。
- `include/gemm/benchmark_utils.h`: 通用 benchmark 工具。
- `include/gemm/kernels.h`: kernel 统一接口。
- `scripts/bench_gemm.sh`: 批量 benchmark 和 CSV 汇总。
- `scripts/visualize_gemm1.py`: 面向教学的 `gemm_1` ASCII 可视化脚本。
- `scripts/visualize_gemm2.py`: 面向教学的 `gemm_2` ASCII 可视化脚本。
- `scripts/compare_gemm1_gemm2_ascii.py`: `gemm_1` 和 `gemm_2` 的并排 ASCII 对比脚本。

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

## ASCII 可视化 `gemm_1`

```bash
python3 scripts/visualize_gemm1.py --m 128 --n 128 --k 16 --thread-x 5 --thread-y 2
```

## ASCII 可视化 `gemm_2`

按线程看：

```bash
python3 scripts/visualize_gemm2.py --m 128 --n 128 --k 16 --thread-x 5 --thread-y 2
```

按 warp 看：

```bash
python3 scripts/visualize_gemm2.py --m 128 --n 128 --k 16 --warp-id 1 --lane 5
```

## 并排对比 `gemm_1` 和 `gemm_2`

比较同一个线程 `(threadIdx.x, threadIdx.y)` 在两种 kernel 里的角色：

```bash
python3 scripts/compare_gemm1_gemm2_ascii.py --thread-x 5 --thread-y 2
```

比较同一个输出 patch 在两种 kernel 中分别由谁负责：

```bash
python3 scripts/compare_gemm1_gemm2_ascii.py --patch-row 1 --patch-col 9
```

这个对比脚本会输出：

- 左右并排的 `gemm_1` / `gemm_2` 线程摘要
- 左右并排的输出 tile 归属图
- 一段总结，说明两者在线程到输出块映射上的关键差异
