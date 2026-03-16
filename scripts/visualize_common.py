#!/usr/bin/env python3
"""Shared helpers for GEMM ASCII visualization scripts."""

from __future__ import annotations

from typing import Sequence


def box(title: str, lines: Sequence[str]) -> None:
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


def zero_accum(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def accumulate_outer_product(
    accum: list[list[float]], a_values: Sequence[float], b_values: Sequence[float]
) -> None:
    for row, a_value in enumerate(a_values):
        for col, b_value in enumerate(b_values):
            accum[row][col] += a_value * b_value


def draw_reg_block(name: str, values: Sequence[float]) -> None:
    print(name)
    print("+--------+--------+--------+--------+")
    print("| " + " | ".join(f"{int(v):6d}" for v in values) + " |")
    print("+--------+--------+--------+--------+")


def draw_accum(accum: Sequence[Sequence[float]], title: str) -> None:
    if not accum:
        return
    cols = len(accum[0])
    print(title)
    print("      " + " ".join(f"c{c}" for c in range(cols)))
    print("      " + "------------" * cols)
    for row, values in enumerate(accum):
        cells = [f"{int(value):10d}" for value in values]
        print(f"r{row} |" + "|".join(cells) + "|")
    print()


def resolve_trace_scope(
    total_k: int, tile_k: int, selected_tile_k: int | None, selected_kk: int | None
) -> tuple[list[int], list[int]]:
    if selected_tile_k is not None:
        if selected_tile_k < 0 or selected_tile_k >= total_k:
            raise SystemExit(f"--tile-k must be in [0, {total_k - 1}].")
        if selected_tile_k % tile_k != 0:
            raise SystemExit(f"--tile-k must be a multiple of {tile_k}.")
        tile_starts = [selected_tile_k]
    else:
        tile_starts = list(range(0, total_k, tile_k))

    if selected_kk is not None:
        if not (0 <= selected_kk < tile_k):
            raise SystemExit(f"--kk must be in [0, {tile_k - 1}].")
        kk_values = [selected_kk]
    else:
        kk_values = list(range(tile_k))

    return tile_starts, kk_values
