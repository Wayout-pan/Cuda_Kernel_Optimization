# Operator Expansion Guide

这份文档定义的是“后续把新算子接入这个仓库时，尽量保持一致的工程约定”。

目标不是一次性做成大而全框架，而是让每个算子模块都能沿着相同的节奏生长：

1. 先有最小可运行版本
2. 再有 benchmark
3. 再逐步增加优化版本
4. 最后视需要补可视化或教学脚本

## 目录约定

新增一个算子 `<op>` 时，优先使用下面的目录布局：

```text
include/<op>/
src/<op>/
scripts/bench_<op>.sh
results/<op>_*.csv
```

如果算子需要额外的教学脚本或分析脚本，可以继续放在 `scripts/` 下，命名建议保持：

```text
scripts/visualize_<op>.py
scripts/visualize_<op>_*.py
scripts/compare_<op>_*.py
```

## 源码布局建议

`src/<op>/` 内建议优先按下面的职责划分：

- `<op>_runner.cu` 或 `<op>_runner.cc`
  - 唯一运行入口
  - 参数解析
  - 输入准备
  - 显存管理
  - 计时和结果汇总
  - 正确性检查

- `kernel_registry.cc`
  - 注册当前算子的所有可运行实现

- `<op>_0.cu`, `<op>_1.cu`, `<op>_2.cu`
  - 从 baseline 到优化版本逐步演进

## 命名建议

- baseline 实现优先使用 `<op>_0`
- 后续优化版本按 `<op>_1`, `<op>_2`, `<op>_3` 递增
- benchmark 脚本优先使用 `bench_<op>.sh`
- 结果文件优先使用 `<op>_bench_<timestamp>.csv`

这样做的好处是：

- 命名简单
- 版本顺序直观
- 便于和当前 GEMM 模块保持一致

## 公共能力的处理方式

如果某些工具明显是跨算子复用的，不要继续塞进某个具体算子目录里。优先考虑抽到更通用的位置，例如未来可以演进出：

- `include/common/`
- `src/common/`
- `scripts/common/`

目前仓库还没有强行抽这一层，是因为现阶段只有 GEMM 模块，过早抽象容易引入空壳结构。等第二个、第三个算子进入仓库后，再根据重复代码做归并更稳妥。

## 新算子接入时的最小清单

建议至少完成下面这些内容：

1. 一个 baseline kernel
2. 一个统一 runner
3. 一个 registry
4. 一个 benchmark 脚本
5. 一段 README 用法说明

如果这个算子的线程映射或 shared memory 设计比较复杂，再额外补：

1. ASCII 可视化脚本
2. 映射对比脚本
3. 教学日志或示例输出

## Makefile 约定

当前 Makefile 已经保留了：

- `make gemm`
- `make list-gemm`
- `make bench-gemm`

后续新增算子时，建议继续增加类似目标：

- `make conv`
- `make list-conv`
- `make bench-conv`

同时保留顶层兼容入口是否需要统一到 `make all` / `make bench`，可以在算子数量更多时再决定。

## 一条实践原则

先保持“模块内自洽”，再考虑“跨模块抽象”。

也就是说：

- 当只有一个算子在使用某段逻辑时，允许它先留在自己的模块里
- 当两个及以上算子开始复制同样的逻辑时，再把它们抽成公共层

这比一开始就设计成复杂框架更适合当前这个仓库的演进节奏。
