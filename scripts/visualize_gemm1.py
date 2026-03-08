#!/usr/bin/env python3
"""ASCII teaching trace for src/gemm/gemm_1.cu."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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
        "GEMM_1 ASCII Visualization",
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


def tile_partition(info: ThreadMapping) -> None:
    print("ASCII map: each cell is one 8x8 output patch, labeled by tid")
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
    box(
        f"K tile starting at k={tile_start_k}",
        [
            f"A load : A[{a_row}, {a_cols[0]}:{a_cols[-1] + 1}] -> s_a[{info.load_a_smem_m}, {info.load_a_smem_k}:{info.load_a_smem_k + 4}]",
            f"B load : B[{b_row}, {b_cols[0]}:{b_cols[-1] + 1}] -> s_b[{info.load_b_smem_k}, {info.load_b_smem_n}:{info.load_b_smem_n + 4}]",
            f"compute : this thread accumulates an 8x8 block starting at local C[{info.comp_a_row0}, {info.comp_b_col0}]",
        ],
    )
    print()


def draw_sa(info: ThreadMapping, base_y: int, tile_start_k: int) -> None:
    row_begin = max(0, info.load_a_smem_m - 2)
    row_end = min(BM, info.load_a_smem_m + 3)
    col_begin = info.load_a_smem_k
    col_end = min(BK, info.load_a_smem_k + 4)
    print("s_a local ASCII window")
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
    print("selected thread writes the bracketed cells")
    print()


def draw_sb(info: ThreadMapping, base_x: int, tile_start_k: int) -> None:
    row_begin = max(0, info.load_b_smem_k)
    row_end = min(BK, info.load_b_smem_k + 1)
    col_begin = max(0, info.load_b_smem_n - 8)
    col_end = min(BN, info.load_b_smem_n + 12)
    print("s_b local ASCII window")
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
    print("selected thread writes the bracketed cells")
    print()


def draw_regs(name: str, values: list[float]) -> None:
    print(name)
    print("+--------+--------+--------+--------+")
    print("| " + " | ".join(f"{int(v):6d}" for v in values) + " |")
    print("+--------+--------+--------+--------+")


def kk_step(args: argparse.Namespace, info: ThreadMapping, tile_start_k: int, kk: int, accum: list[list[float]]) -> None:
    base_x = args.block_x * BN
    base_y = args.block_y * BM
    actual_k = tile_start_k + kk
    a_vec = [make_value_a(base_y + info.comp_a_row0 + d, actual_k) for d in range(TM)]
    b_vec = [make_value_b(actual_k, base_x + info.comp_b_col0 + d) for d in range(TN)]
    for r in range(TM):
        for c in range(TN):
            accum[r][c] += a_vec[r] * b_vec[c]
    box(
        f"kk={kk}  actual_k={actual_k}",
        [
            f"example: c[0,0] += {int(a_vec[0])} * {int(b_vec[0])} -> {int(accum[0][0])}",
            f"example: c[7,7] += {int(a_vec[7])} * {int(b_vec[7])} -> {int(accum[7][7])}",
        ],
    )
    draw_regs("A values used by this thread for kk", a_vec[:4])
    draw_regs("B values used by this thread for kk", b_vec[:4])
    print("remaining A values: " + " ".join(f"{int(v):6d}" for v in a_vec[4:]))
    print("remaining B values: " + " ".join(f"{int(v):6d}" for v in b_vec[4:]))
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
    base_x = args.block_x * BN
    base_y = args.block_y * BM
    print("Final global store ASCII map for selected thread")
    print("      left float4                right float4")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.m % BM != 0 or args.n % BN != 0 or args.k % BK != 0:
        raise SystemExit(f"gemm_1 requires M%{BM}=0, N%{BN}=0, K%{BK}=0. Got M={args.m}, N={args.n}, K={args.k}.")
    if not (0 <= args.thread_x < BLOCK_X and 0 <= args.thread_y < BLOCK_Y):
        raise SystemExit(f"thread-x and thread-y must be in [0,{BLOCK_X - 1}] and [0,{BLOCK_Y - 1}].")

    info = mapping(args.thread_x, args.thread_y)
    header(args, info)
    tile_partition(info)
    for tile_start_k in range(0, args.k, BK):
        load_maps(args, info, tile_start_k)
        draw_sa(info, args.block_y * BM, tile_start_k)
        draw_sb(info, args.block_x * BN, tile_start_k)
        accum = [[0.0 for _ in range(TN)] for _ in range(TM)]
        for kk in range(BK):
            kk_step(args, info, tile_start_k, kk, accum)
        draw_accum(accum)
    store_view(args, info)


if __name__ == "__main__":
    main()
