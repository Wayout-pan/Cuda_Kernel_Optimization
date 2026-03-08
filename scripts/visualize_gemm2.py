#!/usr/bin/env python3
"""ASCII teaching trace for src/gemm/gemm_2.cu."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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


def box(title: str, lines: list[str]) -> None:
    width = max(len(title), *(len(line) for line in lines)) if lines else len(title)
    print("+" + "-" * (width + 2) + "+")
    print(f"| {title.ljust(width)} |")
    print("+" + "-" * (width + 2) + "+")
    for line in lines:
        print(f"| {line.ljust(width)} |")
    print("+" + "-" * (width + 2) + "+")


def make_value_a(row: int, col: int) -> float:
    return row * 100.0 + col


def make_value_b(row: int, col: int) -> float:
    return row * 1000.0 + col


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
    base_x = args.block_x * BLOCK_SIZE * TM
    base_y = args.block_y * BLOCK_SIZE * TN
    box(
        "GEMM_2 ASCII Visualization",
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


def tile_partition(info: ThreadMapping) -> None:
    print("ASCII map: each cell is one 8x8 patch of C, labeled by owner tid")
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
    base_x = args.block_x * BLOCK_SIZE * TM
    base_y = args.block_y * BLOCK_SIZE * TN
    print(f"ASCII map: warp {info.warp_id} ownership inside the 128x128 block tile")
    print("Each cell shows laneXX. Dots mean this 8x8 patch belongs to another warp.")
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
    print("Warp member list")
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
    print("* marks the selected thread/lane")
    print()


def draw_suba(info: ThreadMapping, base_y: int, tile_start_k: int) -> None:
    row_begin = max(0, info.row_a - 2)
    row_end = min(BLOCK_M, info.row_a + 3)
    col_begin = max(0, info.col_a)
    col_end = min(BLOCK_K, info.col_a + 4)
    print("subA local ASCII window")
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
    print("selected thread writes the bracketed cells")
    print()


def draw_subb(info: ThreadMapping, base_x: int, tile_start_k: int) -> None:
    print("subB local ASCII window")
    kk_center = info.tid * 4 // BLOCK_N
    n_center = info.tid * 4 % BLOCK_N
    kk_begin = max(0, kk_center)
    kk_end = min(BLOCK_K, kk_center + 1)
    n_begin = max(0, n_center - 8)
    n_end = min(BLOCK_N, n_center + 12)
    print("      " + " ".join(f"n{n:03d}" for n in range(n_begin, n_end)))
    print("      " + "-------" * (n_end - n_begin))
    for kk in range(kk_begin, kk_end):
        cells = []
        for nn in range(n_begin, n_end):
            value = int(make_value_b(tile_start_k + kk, base_x + nn))
            cell = f"{value:5d}"
            if info.tid * 4 <= kk * BLOCK_N + nn < info.tid * 4 + 4:
                cell = f"[{value:5d}]"
            else:
                cell = f" {value:5d} "
            cells.append(cell)
        print(f"k{kk:02d} |" + "|".join(cells) + "|")
    print("selected thread writes the bracketed cells")
    print()


def k_tile_load_view(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int) -> None:
    base_x = args.block_x * BLOCK_SIZE * TM
    base_y = args.block_y * BLOCK_SIZE * TN
    a_cols = [tile_start_k + info.col_a + t for t in range(4)]
    b_cols = [base_x + info.col_b + t for t in range(4)]
    box(
        f"K tile starting at k={tile_start_k}",
        [
            f"A load : row {base_y + info.row_a}, cols {a_cols}",
            f"B load : row {tile_start_k + info.row_b}, cols {b_cols}",
            f"subA   : rowslot {info.row_a}, colslots {list(range(info.col_a, info.col_a + 4))}",
            f"subB   : linear slots {list(range(info.tid * 4, info.tid * 4 + 4))}",
        ],
    )
    draw_suba(info, base_y, tile_start_k)
    draw_subb(info, base_x, tile_start_k)


def draw_reg_block(name: str, values: list[float]) -> None:
    print(f"{name}")
    print("+--------+--------+--------+--------+")
    print("| " + " | ".join(f"{int(v):6d}" for v in values) + " |")
    print("+--------+--------+--------+--------+")


def kk_step(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int, kk: int, accum: list[list[float]]) -> None:
    base_x = args.block_x * BLOCK_SIZE * TM
    base_y = args.block_y * BLOCK_SIZE * TN
    actual_k = tile_start_k + kk
    reg_a0 = [make_value_a(base_y + info.row_c + d, actual_k) for d in range(4)]
    reg_a1 = [make_value_a(base_y + info.row_c + 4 + d, actual_k) for d in range(4)]
    reg_b0 = [make_value_b(actual_k, base_x + info.col_c + d) for d in range(4)]
    reg_b1 = [make_value_b(actual_k, base_x + info.col_c + 4 + d) for d in range(4)]
    all_a = reg_a0 + reg_a1
    all_b = reg_b0 + reg_b1
    for r in range(TM):
        for c in range(TN):
            accum[r][c] += all_a[r] * all_b[c]
    box(
        f"kk={kk}  actual_k={actual_k}",
        [
            f"selected output patch top-left = ({base_y + info.row_c}, {base_x + info.col_c})",
            f"example: c[0,0] += {int(all_a[0])} * {int(all_b[0])} -> {int(accum[0][0])}",
            f"example: c[7,7] += {int(all_a[7])} * {int(all_b[7])} -> {int(accum[7][7])}",
        ],
    )
    draw_reg_block("regA[0]", reg_a0)
    draw_reg_block("regA[1]", reg_a1)
    draw_reg_block("regB[0]", reg_b0)
    draw_reg_block("regB[1]", reg_b1)
    print()


def draw_accum(accum: list[list[float]]) -> None:
    print("8x8 register accumulator tile")
    print("      " + " ".join(f"c{c}" for c in range(TN)))
    print("      " + "------------" * TN)
    for r in range(TM):
        cells = [f"{int(accum[r][c]):10d}" for c in range(TN)]
        print(f"r{r} |" + "|".join(cells) + "|")
    print()


def store_view(args: argparse.Namespace, info: ThreadMapping) -> None:
    base_x = args.block_x * BLOCK_SIZE * TM
    base_y = args.block_y * BLOCK_SIZE * TN
    print("Final global store ASCII map for selected thread")
    print("      left float4                right float4")
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
    header(args, info)
    tile_partition(info)
    warp_overview(args, info)
    for tile_start_k in range(0, args.k, BLOCK_K):
        k_tile_load_view(args, info, tile_start_k)
        accum = [[0.0 for _ in range(TN)] for _ in range(TM)]
        for kk in range(BLOCK_K):
            kk_step(args, info, tile_start_k, kk, accum)
        draw_accum(accum)
    store_view(args, info)


if __name__ == "__main__":
    main()
