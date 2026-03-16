#!/usr/bin/env python3
"""ASCII teaching trace for src/gemm/gemm_1.cu."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from visualize_common import (
    accumulate_outer_product,
    box,
    draw_accum,
    draw_reg_block,
    make_value_a,
    make_value_b,
    resolve_trace_scope,
    zero_accum,
)

BM = 128
BN = 128
BK = 8
TM = 8
TN = 8
BLOCK_X = BN // TN
BLOCK_Y = BM // TM
THREADS = BLOCK_X * BLOCK_Y


@dataclass
class ThreadMapping:
    tid: int
    tx: int
    ty: int
    load_a_smem_m: int
    load_a_smem_k: int
    load_b_smem_k: int
    load_b_smem_n: int
    comp_a_row0: int
    comp_b_col0: int


def mapping(tx: int, ty: int) -> ThreadMapping:
    tid = ty * BLOCK_X + tx
    load_a_smem_m = tid >> 1
    load_a_smem_k = (tid & 1) << 2
    load_b_smem_k = tid >> 5
    load_b_smem_n = (tid & 31) << 2
    comp_a_row0 = ty * TM
    comp_b_col0 = tx * TN
    return ThreadMapping(tid, tx, ty, load_a_smem_m, load_a_smem_k, load_b_smem_k, load_b_smem_n, comp_a_row0, comp_b_col0)


def header(args: argparse.Namespace, info: ThreadMapping) -> None:
    box(
        "GEMM_1 ASCII 教学可视化",
        [
            f"shape      : M={args.m} N={args.n} K={args.k}",
            f"blockIdx    : ({args.block_x}, {args.block_y})",
            f"threadIdx   : ({args.thread_x}, {args.thread_y})",
            f"tid         : {info.tid}",
            f"block size  : ({BLOCK_X}, {BLOCK_Y}) = {THREADS} threads",
            f"block tile  : C[{args.block_y * BM}:{args.block_y * BM + BM}, {args.block_x * BN}:{args.block_x * BN + BN}]",
            f"thread tile : C[{args.block_y * BM + info.comp_a_row0}:{args.block_y * BM + info.comp_a_row0 + TM}, {args.block_x * BN + info.comp_b_col0}:{args.block_x * BN + info.comp_b_col0 + TN}]",
        ],
    )
    print()


def launch_overview(args: argparse.Namespace) -> None:
    grid_x = (args.n + BN - 1) // BN
    grid_y = (args.m + BM - 1) // BM
    block_row0 = args.block_y * BM
    block_col0 = args.block_x * BN
    box(
        "1. Grid / Block 是怎么分的",
        [
            f"gridDim  = ({grid_x}, {grid_y})，因为每个 block 固定负责 C 的一个 {BM}x{BN} tile",
            f"blockDim = ({BLOCK_X}, {BLOCK_Y})，共 {THREADS} 个线程",
            f"blockIdx.x 沿 N 方向切块，blockIdx.y 沿 M 方向切块",
            f"当前 block ({args.block_x}, {args.block_y}) 覆盖 C[{block_row0}:{block_row0 + BM}, {block_col0}:{block_col0 + BN}]",
            f"对应的 A 行范围是 [{block_row0}:{block_row0 + BM})，B 列范围是 [{block_col0}:{block_col0 + BN})",
            f"K 维按 {BK} 一组推进，因此这次一共要做 {args.k // BK} 轮 K tile",
        ],
    )
    print()


def thread_overview(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_row = args.block_y * BM + info.comp_a_row0
    base_col = args.block_x * BN + info.comp_b_col0
    box(
        "2. 当前线程负责什么",
        [
            f"tid 公式: tid = ty * {BLOCK_X} + tx = {info.ty} * {BLOCK_X} + {info.tx} = {info.tid}",
            f"每个线程累计一个 {TM}x{TN} 的寄存器块，也就是 {TM * TN} 个输出元素",
            f"当前线程的局部输出起点 = ({info.comp_a_row0}, {info.comp_b_col0})",
            f"对应全局输出 patch = C[{base_row}:{base_row + TM}, {base_col}:{base_col + TN}]",
            f"写回时每一行拆成两个 float4：C[row, {base_col}:{base_col + 4}) 和 C[row, {base_col + 4}:{base_col + TN})",
        ],
    )
    print()


def shared_memory_overview(info: ThreadMapping) -> None:
    box(
        "3. shared memory 是怎么装数据的",
        [
            f"s_a 大小 = {BM}x{BK} = {BM * BK} 个 float，s_b 大小 = {BK}x{BN} = {BK * BN} 个 float",
            f"一个 block 里有 {THREADS} 个线程，所以平均每线程每轮 K tile 从 A 搬 4 个 float、从 B 搬 4 个 float",
            f"A 装载公式: load_a_smem_m = tid >> 1 = {info.load_a_smem_m}，load_a_smem_k = (tid & 1) << 2 = {info.load_a_smem_k}",
            f"B 装载公式: load_b_smem_k = tid >> 5 = {info.load_b_smem_k}，load_b_smem_n = (tid & 31) << 2 = {info.load_b_smem_n}",
            "也就是说，这个线程每轮都会各搬一段 float4 到 s_a 和 s_b，随后整个 block 复用这些数据",
        ],
    )
    print()


def compute_overview(info: ThreadMapping) -> None:
    box(
        "4. 线程在 block 内是怎么计算的",
        [
            f"对每个 kk in [0, {BK})，kernel 都会做: r_c[mm][nn] += s_a[ty*{TM}+mm][kk] * s_b[kk][tx*{TN}+nn]",
            f"代入当前线程后就是: r_c[mm][nn] += s_a[{info.ty * TM}+mm][kk] * s_b[kk][{info.tx * TN}+nn]",
            f"所以它覆盖的是局部 C[{info.comp_a_row0}+mm][{info.comp_b_col0}+nn]，也就是一个 {TM}x{TN} patch",
            "__syncthreads() 的作用是先等所有线程把 s_a/s_b 填满，再一起开始算；本轮算完后再同步，避免下一轮覆盖旧数据",
        ],
    )
    print()


def trace_scope(tile_starts: list[int], kk_values: list[int], summary_only: bool) -> None:
    tile_desc = ", ".join(f"[{start}:{start + BK})" for start in tile_starts)
    kk_desc = ", ".join(str(kk) for kk in kk_values)
    box(
        "5. 这次脚本会展示哪些步骤",
        [
            f"会展开的 K tile: {tile_desc}",
            f"会展开的 kk    : {kk_desc}",
            f"详细程度       : {'只看总览和最终寄存器结果' if summary_only else '把每个选中的 tile / kk 都展开'}",
            "无论是否裁剪显示，最后都会给出跨完整 K 维度的最终寄存器累加结果",
        ],
    )
    print()


def tile_partition(info: ThreadMapping) -> None:
    print("ASCII 图：每个格子都是一个 8x8 输出 patch，格子里的数字是负责它的 tid")
    print("     " + " ".join(f"c{c:02d}" for c in range(BLOCK_X)))
    print("     " + "-----" * BLOCK_X)
    for pr in range(BLOCK_Y):
        cells = []
        for pc in range(BLOCK_X):
            owner = pr * BLOCK_X + pc
            if owner == info.tid:
                cells.append(f"[{owner:03d}]")
            else:
                cells.append(f" {owner:03d} ")
        print(f"r{pr:02d} |" + "|".join(cells) + "|")
    print()


def load_maps(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int) -> None:
    base_x = args.block_x * BN
    base_y = args.block_y * BM
    a_row = base_y + info.load_a_smem_m
    a_cols = [tile_start_k + info.load_a_smem_k + d for d in range(4)]
    b_row = tile_start_k + info.load_b_smem_k
    b_cols = [base_x + info.load_b_smem_n + d for d in range(4)]
    a_offset = a_row * args.k + a_cols[0]
    b_offset = b_row * args.n + b_cols[0]
    box(
        f"当前 K tile：k=[{tile_start_k}:{tile_start_k + BK})",
        [
            f"A 装载: A[{a_row}, {a_cols[0]}:{a_cols[-1] + 1}] -> s_a[{info.load_a_smem_m}, {info.load_a_smem_k}:{info.load_a_smem_k + 4}]",
            f"A 偏移: OFFSET({a_row}, {a_cols[0]}, ld={args.k}) = {a_row} * {args.k} + {a_cols[0]} = {a_offset}",
            f"B 装载: B[{b_row}, {b_cols[0]}:{b_cols[-1] + 1}] -> s_b[{info.load_b_smem_k}, {info.load_b_smem_n}:{info.load_b_smem_n + 4}]",
            f"B 偏移: OFFSET({b_row}, {b_cols[0]}, ld={args.n}) = {b_row} * {args.n} + {b_cols[0]} = {b_offset}",
            f"这一轮计算时，本线程会继续累计局部 C[{info.comp_a_row0}:{info.comp_a_row0 + TM}, {info.comp_b_col0}:{info.comp_b_col0 + TN}]",
        ],
    )
    print()


def draw_sa(info: ThreadMapping, base_y: int, tile_start_k: int) -> None:
    row_begin = max(0, info.load_a_smem_m - 2)
    row_end = min(BM, info.load_a_smem_m + 3)
    col_begin = info.load_a_smem_k
    col_end = min(BK, info.load_a_smem_k + 4)
    print("s_a 局部窗口")
    print("      " + " ".join(f"k{c:02d}" for c in range(col_begin, col_end)))
    print("      " + "------" * (col_end - col_begin))
    for r in range(row_begin, row_end):
        cells = []
        for c in range(col_begin, col_end):
            value = int(make_value_a(base_y + r, tile_start_k + c))
            if r == info.load_a_smem_m and info.load_a_smem_k <= c < info.load_a_smem_k + 4:
                cells.append(f"[{value:4d}]")
            else:
                cells.append(f" {value:4d} ")
        print(f"r{r:03d} |" + "|".join(cells) + "|")
    print("带 [] 的格子就是当前线程写入 shared memory 的位置")
    print()


def draw_sb(info: ThreadMapping, base_x: int, tile_start_k: int) -> None:
    row_begin = max(0, info.load_b_smem_k)
    row_end = min(BK, info.load_b_smem_k + 1)
    col_begin = max(0, info.load_b_smem_n - 8)
    col_end = min(BN, info.load_b_smem_n + 12)
    print("s_b 局部窗口")
    print("      " + " ".join(f"n{c:03d}" for c in range(col_begin, col_end)))
    print("      " + "-------" * (col_end - col_begin))
    for r in range(row_begin, row_end):
        cells = []
        for c in range(col_begin, col_end):
            value = int(make_value_b(tile_start_k + r, base_x + c))
            if r == info.load_b_smem_k and info.load_b_smem_n <= c < info.load_b_smem_n + 4:
                cells.append(f"[{value:5d}]")
            else:
                cells.append(f" {value:5d} ")
        print(f"k{r:02d} |" + "|".join(cells) + "|")
    print("带 [] 的格子就是当前线程写入 shared memory 的位置")
    print()


def kk_step(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int, kk: int, accum: list[list[float]]) -> None:
    base_x = args.block_x * BN
    base_y = args.block_y * BM
    actual_k = tile_start_k + kk
    a_vec = [make_value_a(base_y + info.comp_a_row0 + d, actual_k) for d in range(TM)]
    b_vec = [make_value_b(actual_k, base_x + info.comp_b_col0 + d) for d in range(TN)]
    accumulate_outer_product(accum, a_vec, b_vec)
    box(
        f"kk={kk}  actual_k={actual_k}",
        [
            f"本轮实际使用的 K 下标 = tile_start_k + kk = {tile_start_k} + {kk} = {actual_k}",
            f"示例 1: c[0,0] += {int(a_vec[0])} * {int(b_vec[0])} -> {int(accum[0][0])}",
            f"示例 2: c[7,7] += {int(a_vec[7])} * {int(b_vec[7])} -> {int(accum[7][7])}",
        ],
    )
    draw_reg_block("A 向量 [0:4]", a_vec[:4])
    draw_reg_block("A 向量 [4:8]", a_vec[4:])
    draw_reg_block("B 向量 [0:4]", b_vec[:4])
    draw_reg_block("B 向量 [4:8]", b_vec[4:])
    print()


def build_full_accum(args: argparse.Namespace, info: ThreadMapping) -> list[list[float]]:
    base_x = args.block_x * BN
    base_y = args.block_y * BM
    accum = zero_accum(TM, TN)
    for actual_k in range(args.k):
        a_vec = [make_value_a(base_y + info.comp_a_row0 + d, actual_k) for d in range(TM)]
        b_vec = [make_value_b(actual_k, base_x + info.comp_b_col0 + d) for d in range(TN)]
        accumulate_outer_product(accum, a_vec, b_vec)
    return accum


def store_view(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BN
    base_y = args.block_y * BM
    print("最终写回 global memory 的位置")
    print("      左侧 float4                右侧 float4")
    print("+----+------------------------+------------------------+")
    for i in range(TM):
        row = base_y + info.comp_a_row0 + i
        left0 = base_x + info.comp_b_col0
        right0 = base_x + info.comp_b_col0 + 4
        print(f"| r{i} | C[{row:3d},{left0:3d}:{left0 + 4:3d}]        | C[{row:3d},{right0:3d}:{right0 + 4:3d}]        |")
    print("+----+------------------------+------------------------+")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the logical execution of gemm_1 for one block/thread")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--block-x", type=int, default=0)
    parser.add_argument("--block-y", type=int, default=0)
    parser.add_argument("--thread-x", type=int, default=0)
    parser.add_argument("--thread-y", type=int, default=0)
    parser.add_argument("--tile-k", type=int, help="Only render the K tile starting at this global k offset")
    parser.add_argument("--kk", type=int, help="Only render this kk inside each shown K tile")
    parser.add_argument("--summary-only", action="store_true", help="Skip per-step tile details and only print mapping plus final accumulation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.m % BM != 0 or args.n % BN != 0 or args.k % BK != 0:
        raise SystemExit(f"gemm_1 requires M%{BM}=0, N%{BN}=0, K%{BK}=0. Got M={args.m}, N={args.n}, K={args.k}.")
    if not (0 <= args.thread_x < BLOCK_X and 0 <= args.thread_y < BLOCK_Y):
        raise SystemExit(f"thread-x and thread-y must be in [0,{BLOCK_X - 1}] and [0,{BLOCK_Y - 1}].")

    info = mapping(args.thread_x, args.thread_y)
    tile_starts, kk_values = resolve_trace_scope(args.k, BK, args.tile_k, args.kk)
    full_accum = build_full_accum(args, info)

    header(args, info)
    launch_overview(args)
    thread_overview(args, info)
    shared_memory_overview(info)
    compute_overview(info)
    trace_scope(tile_starts, kk_values, args.summary_only)
    tile_partition(info)

    if not args.summary_only:
        for tile_start_k in tile_starts:
            load_maps(args, info, tile_start_k)
            draw_sa(info, args.block_y * BM, tile_start_k)
            draw_sb(info, args.block_x * BN, tile_start_k)
            visible_accum = zero_accum(TM, TN)
            for kk in kk_values:
                kk_step(args, info, tile_start_k, kk, visible_accum)
            draw_accum(visible_accum, f"当前可见步骤累计出的寄存器块（只统计 K tile [{tile_start_k}:{tile_start_k + BK}) 里选中的 kk）")

    draw_accum(full_accum, "跨完整 K 维度后的最终寄存器累加结果")
    store_view(args, info)


if __name__ == "__main__":
    main()
