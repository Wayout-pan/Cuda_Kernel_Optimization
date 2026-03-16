#!/usr/bin/env python3
"""ASCII teaching trace for src/gemm/gemm_2.cu."""

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

BLOCK_SIZE = 16
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 8
TM = BLOCK_M // BLOCK_SIZE
TN = BLOCK_N // BLOCK_SIZE
WARP_SIZE = 32
WARPS_PER_BLOCK = (BLOCK_SIZE * BLOCK_SIZE) // WARP_SIZE


@dataclass
class ThreadMapping:
    tid: int
    warp_id: int
    warp_lane: int
    row_a: int
    row_b: int
    col_a: int
    col_b: int
    row_c: int
    col_c: int


def mapping(thread_x: int, thread_y: int) -> ThreadMapping:
    tid = thread_y * BLOCK_SIZE + thread_x
    warp_id = tid >> 5
    warp_lane = tid & 31
    row_a = tid >> 1
    row_b = tid >> 5
    col_a = (tid & 1) << 2
    col_b = (tid << 2) & 127
    row_c = ((warp_id >> 1 << 2) + (warp_lane & 3)) << 3
    col_c = (((warp_id & 1) << 3) + (warp_lane >> 2)) << 3
    return ThreadMapping(tid, warp_id, warp_lane, row_a, row_b, col_a, col_b, row_c, col_c)


def mapping_from_tid(tid: int) -> ThreadMapping:
    return mapping(tid % BLOCK_SIZE, tid // BLOCK_SIZE)


def pick_thread_from_warp(warp_id: int, lane: int) -> tuple[int, int]:
    tid = warp_id * WARP_SIZE + lane
    return tid % BLOCK_SIZE, tid // BLOCK_SIZE


def header(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    box(
        "GEMM_2 ASCII 教学可视化",
        [
            f"shape      : M={args.m} N={args.n} K={args.k}",
            f"blockIdx    : ({args.block_x}, {args.block_y})",
            f"threadIdx   : ({args.thread_x}, {args.thread_y})",
            f"tid         : {info.tid}",
            f"warp/lane   : warp {info.warp_id}, lane {info.warp_lane}",
            f"block tile  : C[{base_y}:{base_y + BLOCK_M}, {base_x}:{base_x + BLOCK_N}]",
            f"thread tile : C[{base_y + info.row_c}:{base_y + info.row_c + TM}, {base_x + info.col_c}:{base_x + info.col_c + TN}]",
            "note        : logical index trace, not hardware cycle trace",
        ],
    )
    print()


def launch_overview(args: argparse.Namespace) -> None:
    grid_x = (args.n + BLOCK_N - 1) // BLOCK_N
    grid_y = (args.m + BLOCK_M - 1) // BLOCK_M
    block_row0 = args.block_y * BLOCK_M
    block_col0 = args.block_x * BLOCK_N
    box(
        "1. Grid / Block 是怎么分的",
        [
            f"gridDim  = ({grid_x}, {grid_y})，每个 block 固定负责 C 的一个 {BLOCK_M}x{BLOCK_N} tile",
            f"blockDim = ({BLOCK_SIZE}, {BLOCK_SIZE})，共 {BLOCK_SIZE * BLOCK_SIZE} 个线程",
            f"blockIdx.x 沿 N 方向切块，blockIdx.y 沿 M 方向切块",
            f"当前 block ({args.block_x}, {args.block_y}) 覆盖 C[{block_row0}:{block_row0 + BLOCK_M}, {block_col0}:{block_col0 + BLOCK_N}]",
            f"K 维按 {BLOCK_K} 一组推进，因此这次一共要做 {args.k // BLOCK_K} 轮 K tile",
        ],
    )
    print()


def thread_overview(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    warp_tid0 = info.warp_id * WARP_SIZE
    warp_members = [mapping_from_tid(warp_tid0 + lane) for lane in range(WARP_SIZE)]
    warp_row0 = min(member.row_c for member in warp_members)
    warp_row1 = max(member.row_c for member in warp_members) + TM
    warp_col0 = min(member.col_c for member in warp_members)
    warp_col1 = max(member.col_c for member in warp_members) + TN
    box(
        "2. 当前线程 / warp 负责什么",
        [
            f"tid 公式: tid = ty * {BLOCK_SIZE} + tx = {args.thread_y} * {BLOCK_SIZE} + {args.thread_x} = {info.tid}",
            f"warp 划分: warp_id = tid >> 5 = {info.warp_id}，warp_lane = tid & 31 = {info.warp_lane}",
            f"rowC = ((warp_id >> 1 << 2) + (lane & 3)) << 3 = {info.row_c}",
            f"colC = (((warp_id & 1) << 3) + (lane >> 2)) << 3 = {info.col_c}",
            f"当前线程最终负责 C[{base_y + info.row_c}:{base_y + info.row_c + TM}, {base_x + info.col_c}:{base_x + info.col_c + TN}]",
            f"同一个 warp 的 32 个线程会被重排后覆盖 block 内的 C[{base_y + warp_row0}:{base_y + warp_row1}, {base_x + warp_col0}:{base_x + warp_col1}]",
        ],
    )
    print()


def shared_memory_overview(info: ThreadMapping) -> None:
    box(
        "3. shared memory 是怎么装数据的",
        [
            f"subA 逻辑上是 {BLOCK_M}x{BLOCK_K}，subB 逻辑上是 {BLOCK_K}x{BLOCK_N}",
            f"A 装载索引: rowA = tid >> 1 = {info.row_a}，colA = (tid & 1) << 2 = {info.col_a}",
            f"B 装载索引: rowB = tid >> 5 = {info.row_b}，colB = (tid << 2) & 127 = {info.col_b}",
            "B 的写法比较直接：subB[tid*4 : tid*4+4] = float4(B[...])，等价于把一个 8x128 tile 线性铺平后顺序写入",
            f"A 的写法更特别：subA[rowA + colA*{BLOCK_M}] 这种“按列存”的方式，相当于把 A tile 转成更利于后续读取的布局",
            "这样做的目的，是让后面的 regA / regB 可以按 float4 连续读出来，减少地址计算并改善访问模式",
        ],
    )
    print()


def compute_overview(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    box(
        "4. 线程在 block 内是怎么计算的",
        [
            f"每个 kk 都会从 shared memory 取出 8 个 A 值和 8 个 B 值，对应 patch 左上角 ({base_y + info.row_c}, {base_x + info.col_c})",
            f"regA[0]/regA[1] 组成 A[{base_y + info.row_c}:{base_y + info.row_c + TM}, k]，regB[0]/regB[1] 组成 B[k, {base_x + info.col_c}:{base_x + info.col_c + TN}]",
            f"随后在线程私有寄存器里做一个 {TM}x{TN} 的 outer product，累加到 c[64]",
            "这就是 gemm_2 比 gemm_1 更激进的地方：输出 patch 的线程归属被 warp 级重排，计算也被手工展开到了寄存器级别",
        ],
    )
    print()


def trace_scope(tile_starts: list[int], kk_values: list[int], summary_only: bool) -> None:
    tile_desc = ", ".join(f"[{start}:{start + BLOCK_K})" for start in tile_starts)
    kk_desc = ", ".join(str(kk) for kk in kk_values)
    box(
        "5. 这次脚本会展示哪些步骤",
        [
            f"会展开的 K tile: {tile_desc}",
            f"会展开的 kk    : {kk_desc}",
            f"详细程度       : {'只看总览、warp 归属和最终寄存器结果' if summary_only else '把每个选中的 tile / kk 都展开'}",
            "无论是否裁剪显示，最后都会给出跨完整 K 维度的最终寄存器累加结果",
        ],
    )
    print()


def tile_partition(info: ThreadMapping) -> None:
    print("ASCII 图：每个格子都是一个 8x8 输出 patch，格子里的数字是负责它的 tid")
    header_cols = "     " + " ".join(f"c{c:02d}" for c in range(BLOCK_N // TN))
    print(header_cols)
    print("     " + "-----" * (BLOCK_N // TN))
    for patch_row in range(BLOCK_M // TM):
        cells = []
        for patch_col in range(BLOCK_N // TN):
            owner = None
            for tid in range(BLOCK_SIZE * BLOCK_SIZE):
                t = mapping_from_tid(tid)
                if t.row_c // TM == patch_row and t.col_c // TN == patch_col:
                    owner = tid
                    break
            if owner == info.tid:
                cells.append(f"[{owner:03d}]")
            else:
                cells.append(f" {owner:03d} ")
        print(f"r{patch_row:02d} |" + "|".join(cells) + "|")
    print()


def warp_overview(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    print(f"ASCII 图：warp {info.warp_id} 在当前 128x128 block tile 里的 patch 归属")
    print("每个格子显示的是 laneXX；出现 .. 说明这个 8x8 patch 属于别的 warp。")
    print("     " + " ".join(f"c{c:02d}" for c in range(BLOCK_N // TN)))
    print("     " + "-------" * (BLOCK_N // TN))
    for patch_row in range(BLOCK_M // TM):
        cells = []
        for patch_col in range(BLOCK_N // TN):
            label = "  ..  "
            for lane in range(WARP_SIZE):
                tid = info.warp_id * WARP_SIZE + lane
                t = mapping_from_tid(tid)
                if t.row_c // TM == patch_row and t.col_c // TN == patch_col:
                    label = f"L{lane:02d}"
                    if tid == info.tid:
                        label = f"[{label}]"
                    else:
                        label = f" {label} "
                    break
            cells.append(label)
        print(f"r{patch_row:02d} |" + "|".join(cells) + "|")
    print()
    print("当前 warp 的成员分工表")
    print("+------+-----------+-------------+-------------+----------------+")
    print("| lane | threadIdx  | A load      | B load      | C patch start  |")
    print("+------+-----------+-------------+-------------+----------------+")
    for lane in range(WARP_SIZE):
        tid = info.warp_id * WARP_SIZE + lane
        t = mapping_from_tid(tid)
        tx = tid % BLOCK_SIZE
        ty = tid // BLOCK_SIZE
        mark = "*" if tid == info.tid else " "
        print(
            f"| {mark}{lane:02d}  | ({tx:2d},{ty:2d})    | A[{base_y + t.row_a:3d},{t.col_a:1d}:{t.col_a + 4:1d}] |"
            f" B[{t.row_b:1d},{base_x + t.col_b:3d}:{base_x + t.col_b + 4:3d}] | ({base_y + t.row_c:3d},{base_x + t.col_c:3d})         |"
        )
    print("+------+-----------+-------------+-------------+----------------+")
    print("* 这一行表示当前选中的线程 / lane")
    print()


def draw_suba(info: ThreadMapping, base_y: int, tile_start_k: int) -> None:
    row_begin = max(0, info.row_a - 2)
    row_end = min(BLOCK_M, info.row_a + 3)
    col_begin = max(0, info.col_a)
    col_end = min(BLOCK_K, info.col_a + 4)
    print("subA 局部窗口")
    print("      " + " ".join(f"k{c:02d}" for c in range(col_begin, col_end)))
    print("      " + "------" * (col_end - col_begin))
    for r in range(row_begin, row_end):
        cells = []
        for c in range(col_begin, col_end):
            value = int(make_value_a(base_y + r, tile_start_k + c))
            cell = f"{value:4d}"
            if r == info.row_a and info.col_a <= c < info.col_a + 4:
                cell = f"[{value:4d}]"
            else:
                cell = f" {value:4d} "
            cells.append(cell)
        print(f"r{r:03d} |" + "|".join(cells) + "|")
    print("带 [] 的格子就是当前线程写入 shared memory 的位置")
    print()


def draw_subb(info: ThreadMapping, base_x: int, tile_start_k: int) -> None:
    print("subB 局部窗口")
    kk_begin = max(0, info.row_b)
    kk_end = min(BLOCK_K, info.row_b + 1)
    n_begin = max(0, info.col_b - 8)
    n_end = min(BLOCK_N, info.col_b + 12)
    print("      " + " ".join(f"n{n:03d}" for n in range(n_begin, n_end)))
    print("      " + "-------" * (n_end - n_begin))
    for kk in range(kk_begin, kk_end):
        cells = []
        for nn in range(n_begin, n_end):
            value = int(make_value_b(tile_start_k + kk, base_x + nn))
            cell = f"{value:5d}"
            if kk == info.row_b and info.col_b <= nn < info.col_b + 4:
                cell = f"[{value:5d}]"
            else:
                cell = f" {value:5d} "
            cells.append(cell)
        print(f"k{kk:02d} |" + "|".join(cells) + "|")
    print("带 [] 的格子就是当前线程写入 shared memory 的位置")
    print()


def k_tile_load_view(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    a_cols = [tile_start_k + info.col_a + t for t in range(4)]
    b_cols = [base_x + info.col_b + t for t in range(4)]
    a_offset = (base_y + info.row_a) * args.k + a_cols[0]
    b_offset = (tile_start_k + info.row_b) * args.n + b_cols[0]
    box(
        f"当前 K tile：k=[{tile_start_k}:{tile_start_k + BLOCK_K})",
        [
            f"A 装载: A[{base_y + info.row_a}, {a_cols[0]}:{a_cols[-1] + 1}]，起始偏移 = {a_offset}",
            f"B 装载: B[{tile_start_k + info.row_b}, {b_cols[0]}:{b_cols[-1] + 1}]，起始偏移 = {b_offset}",
            f"subA   : rowslot {info.row_a}, colslots {list(range(info.col_a, info.col_a + 4))}",
            (
                f"subB   : rowslot {info.row_b}, colslots {list(range(info.col_b, info.col_b + 4))}，"
                f"线性槽位 {list(range(info.tid * 4, info.tid * 4 + 4))}"
            ),
        ],
    )
    draw_suba(info, base_y, tile_start_k)
    draw_subb(info, base_x, tile_start_k)


def kk_step(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int, kk: int, accum: list[list[float]]) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    actual_k = tile_start_k + kk
    reg_a0 = [make_value_a(base_y + info.row_c + d, actual_k) for d in range(4)]
    reg_a1 = [make_value_a(base_y + info.row_c + 4 + d, actual_k) for d in range(4)]
    reg_b0 = [make_value_b(actual_k, base_x + info.col_c + d) for d in range(4)]
    reg_b1 = [make_value_b(actual_k, base_x + info.col_c + 4 + d) for d in range(4)]
    all_a = reg_a0 + reg_a1
    all_b = reg_b0 + reg_b1
    accumulate_outer_product(accum, all_a, all_b)
    box(
        f"kk={kk}  actual_k={actual_k}",
        [
            f"当前输出 patch 左上角 = ({base_y + info.row_c}, {base_x + info.col_c})",
            f"本轮实际使用的 K 下标 = tile_start_k + kk = {tile_start_k} + {kk} = {actual_k}",
            f"示例 1: c[0,0] += {int(all_a[0])} * {int(all_b[0])} -> {int(accum[0][0])}",
            f"示例 2: c[7,7] += {int(all_a[7])} * {int(all_b[7])} -> {int(accum[7][7])}",
        ],
    )
    draw_reg_block("regA[0]", reg_a0)
    draw_reg_block("regA[1]", reg_a1)
    draw_reg_block("regB[0]", reg_b0)
    draw_reg_block("regB[1]", reg_b1)
    print()


def build_full_accum(args: argparse.Namespace, info: ThreadMapping) -> list[list[float]]:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    accum = zero_accum(TM, TN)
    for actual_k in range(args.k):
        all_a = [make_value_a(base_y + info.row_c + d, actual_k) for d in range(TM)]
        all_b = [make_value_b(actual_k, base_x + info.col_c + d) for d in range(TN)]
        accumulate_outer_product(accum, all_a, all_b)
    return accum


def store_view(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BLOCK_N
    base_y = args.block_y * BLOCK_M
    print("最终写回 global memory 的位置")
    print("      左侧 float4                右侧 float4")
    print("+----+------------------------+------------------------+")
    for i in range(TM):
        row = base_y + info.row_c + i
        left0 = base_x + info.col_c
        right0 = base_x + info.col_c + 4
        print(f"| r{i} | C[{row:3d},{left0:3d}:{left0 + 4:3d}]        | C[{row:3d},{right0:3d}:{right0 + 4:3d}]        |")
    print("+----+------------------------+------------------------+")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the logical execution of gemm_2 for one block/thread")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--block-x", type=int, default=0)
    parser.add_argument("--block-y", type=int, default=0)
    parser.add_argument("--thread-x", type=int, default=0)
    parser.add_argument("--thread-y", type=int, default=0)
    parser.add_argument("--warp-id", type=int, help="Select a warp directly; thread will default to lane 0 unless --lane is given")
    parser.add_argument("--lane", type=int, default=0, help="Lane within the selected warp when --warp-id is used")
    parser.add_argument("--tile-k", type=int, help="Only render the K tile starting at this global k offset")
    parser.add_argument("--kk", type=int, help="Only render this kk inside each shown K tile")
    parser.add_argument("--summary-only", action="store_true", help="Skip per-step tile details and only print mapping plus final accumulation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.m % BLOCK_M != 0 or args.n % BLOCK_N != 0 or args.k % BLOCK_K != 0:
        raise SystemExit(
            f"gemm_2 requires M%{BLOCK_M}=0, N%{BLOCK_N}=0, K%{BLOCK_K}=0. Got M={args.m}, N={args.n}, K={args.k}."
        )
    if args.warp_id is not None:
        if not (0 <= args.warp_id < WARPS_PER_BLOCK):
            raise SystemExit(f"warp-id must be in [0, {WARPS_PER_BLOCK - 1}].")
        if not (0 <= args.lane < WARP_SIZE):
            raise SystemExit(f"lane must be in [0, {WARP_SIZE - 1}].")
        args.thread_x, args.thread_y = pick_thread_from_warp(args.warp_id, args.lane)
    if not (0 <= args.thread_x < BLOCK_SIZE and 0 <= args.thread_y < BLOCK_SIZE):
        raise SystemExit("thread-x and thread-y must be in [0, 15].")

    info = mapping(args.thread_x, args.thread_y)
    tile_starts, kk_values = resolve_trace_scope(args.k, BLOCK_K, args.tile_k, args.kk)
    full_accum = build_full_accum(args, info)

    header(args, info)
    launch_overview(args)
    thread_overview(args, info)
    shared_memory_overview(info)
    compute_overview(args, info)
    trace_scope(tile_starts, kk_values, args.summary_only)
    tile_partition(info)
    warp_overview(args, info)

    if not args.summary_only:
        for tile_start_k in tile_starts:
            k_tile_load_view(args, info, tile_start_k)
            visible_accum = zero_accum(TM, TN)
            for kk in kk_values:
                kk_step(args, info, tile_start_k, kk, visible_accum)
            draw_accum(visible_accum, f"当前可见步骤累计出的寄存器块（只统计 K tile [{tile_start_k}:{tile_start_k + BLOCK_K}) 里选中的 kk）")

    draw_accum(full_accum, "跨完整 K 维度后的最终寄存器累加结果")
    store_view(args, info)


if __name__ == "__main__":
    main()
