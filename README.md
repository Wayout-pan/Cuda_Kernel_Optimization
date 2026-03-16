# CUDA Operator Optimization Playground

这个仓库用于做 CUDA 算子优化实验与教学分析。

当前已经实现的是两个模块：

- 单精度 GEMM (`C = A x B`)
- 一维 sum reduction

后续会逐步扩展到更多算子，例如卷积、归约族算子、归一化、注意力相关 kernel 等。仓库的设计目标不再是“只优化 GEMM”，而是沉淀一套可复用的：

- 算子实现路径
- benchmark 与结果记录方式
- 可视化 / 教学脚本
- 扩展新算子的目录约定

## 当前状态

- 已落地模块：`gemm`、`reduction`
- 已提供能力：kernel 实现、统一 runner、批量 benchmark、ASCII 教学可视化
- 当前默认构建目标会编译所有已落地模块，后续新增算子时会按同样的模块化方式接入

## 仓库结构

- `src/gemm/`: 当前 GEMM 模块的源码目录，包含 kernel、runner 和注册表
- `include/gemm/`: 当前 GEMM 模块的公共头文件
- `src/reduction/`: 当前 reduction 模块的源码目录，包含 kernel、runner 和注册表
- `include/reduction/`: 当前 reduction 模块的公共头文件
- `scripts/bench_gemm.sh`: GEMM 的批量 benchmark 脚本
- `scripts/bench_reduction.sh`: reduction 的批量 benchmark 脚本
- `scripts/visualize_gemm1.py`: `gemm_1` 的 ASCII 教学可视化脚本
- `scripts/visualize_gemm2.py`: `gemm_2` 的 ASCII 教学可视化脚本
- `scripts/compare_gemm1_gemm2_ascii.py`: `gemm_1` / `gemm_2` 的并排对比脚本
- `results/`: benchmark 输出目录
- `docs/operator_expansion.md`: 新算子接入约定与目录规范

## 扩展约定

后续新增算子时，建议遵循下面的布局：

- `src/<op>/`: 算子实现、runner、registry
- `include/<op>/`: 算子对外接口与公共工具
- `scripts/bench_<op>.sh`: 算子的批量 benchmark 脚本
- `scripts/visualize_<op>*.py`: 可选的教学 / 调试脚本
- `results/<op>_*.csv`: 算子 benchmark 结果

这样做的目的是让每个算子模块相对独立，同时保留统一的工程风格。详细约定见 [docs/operator_expansion.md](docs/operator_expansion.md)。

## 当前 GEMM 模块

GEMM 仍然是当前仓库里最完整的示例模块。

- `src/gemm/gemm_*.cu`: 各个 GEMM kernel 的实现、launch 函数和 shape 约束
- `src/gemm/gemm_runner.cu`: GEMM 的统一运行入口，负责参数解析、显存管理、计时和结果校验
- `src/gemm/kernel_registry.cc`: 注册当前所有可用 GEMM kernel
- `include/gemm/benchmark_utils.h`: GEMM 的 benchmark 公共工具
- `include/gemm/kernels.h`: GEMM kernel 统一接口

## 构建与运行

编译当前默认模块：

```bash
make all
```

这会编译当前所有已落地模块。

如果只编译 GEMM：

```bash
make gemm
```

如果只编译 reduction：

```bash
make reduction
```

查看当前所有模块里可用的 kernel：

```bash
make list
```

只看当前 GEMM 模块里可用的 kernel：

```bash
make list-gemm
```

只看当前 reduction 模块里可用的 kernel：

```bash
make list-reduction
```

单独运行某个 GEMM kernel：

```bash
./build/gemm_runner --kernel gemm_1 --m 1024 --n 1024 --k 1024 --warmup 5 --iters 20 --check
```

批量 benchmark 当前所有已落地模块：

```bash
make bench
```

只 benchmark GEMM：

```bash
make bench-gemm
```

只 benchmark reduction：

```bash
make bench-reduction
```

自定义 GEMM case：

```bash
CASES="512x512x512 1024x2048x1024" WARMUP=5 ITERS=20 VERIFY=1 bash scripts/bench_gemm.sh
```

## 当前 Reduction 模块

Reduction 目前实现的是一维 `sum reduction`，用于覆盖 GEMM 之外更典型的 memory-bound 优化路径。

- `src/reduction/reduction_0.cu`: 单 block baseline reduction
- `src/reduction/reduction_1.cu`: 多 block shared-memory reduction，block 间通过 `atomicAdd` 聚合
- `src/reduction/reduction_runner.cu`: reduction 的统一运行入口
- `src/reduction/kernel_registry.cc`: 注册当前所有 reduction kernel
- `include/reduction/benchmark_utils.h`: reduction 的 benchmark 公共工具
- `include/reduction/kernels.h`: reduction kernel 统一接口

单独运行某个 reduction kernel：

```bash
./build/reduction_runner --kernel reduction_1 --n 16777216 --warmup 5 --iters 20 --check
```

批量 benchmark reduction：

```bash
bash scripts/bench_reduction.sh
```

自定义 reduction case：

```bash
CASES="1048576 4194304 16777216" WARMUP=5 ITERS=20 VERIFY=1 bash scripts/bench_reduction.sh
```

## ASCII 可视化 `gemm_1`

```bash
python3 scripts/visualize_gemm1.py --m 128 --n 128 --k 16 --thread-x 5 --thread-y 2
```

只看某个 `K tile` 内的单个 `kk`：

```bash
python3 scripts/visualize_gemm1.py --m 128 --n 128 --k 16 --thread-x 5 --thread-y 2 --tile-k 8 --kk 3
```

只看线程映射和最终寄存器结果：

```bash
python3 scripts/visualize_gemm1.py --m 128 --n 128 --k 16 --thread-x 5 --thread-y 2 --summary-only
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

只看某个 `K tile` 内的单个 `kk`：

```bash
python3 scripts/visualize_gemm2.py --m 128 --n 128 --k 16 --warp-id 1 --lane 5 --tile-k 8 --kk 3
```

只看 warp/线程映射和最终寄存器结果：

```bash
python3 scripts/visualize_gemm2.py --m 128 --n 128 --k 16 --warp-id 1 --lane 5 --summary-only
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
