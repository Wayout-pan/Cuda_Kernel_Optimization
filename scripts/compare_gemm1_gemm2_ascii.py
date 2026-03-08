#!/usr/bin/env python3
"""Side-by-side ASCII comparison for gemm_1 and gemm_2."""

from __future__ import annotations

import argparse

BM = BN = 128
BK = 8
TM = TN = 8
BLOCK = 16
WARP_SIZE = 32


def g1_mapping(tx: int, ty: int) -> dict[str, int]:
    tid = ty * BLOCK + tx
    return {
        "tid": tid,
        "tx": tx,
        "ty": ty,
        "load_a_row": tid >> 1,
        "load_a_col": (tid & 1) << 2,
        "load_b_row": tid >> 5,
        "load_b_col": (tid & 31) << 2,
        "c_row": ty * TM,
        "c_col": tx * TN,
    }


def g2_mapping(tx: int, ty: int) -> dict[str, int]:
    tid = ty * BLOCK + tx
    warp_id = tid >> 5
    warp_lane = tid & 31
    return {
        "tid": tid,
        "tx": tx,
        "ty": ty,
        "warp_id": warp_id,
        "warp_lane": warp_lane,
        "load_a_row": tid >> 1,
        "load_a_col": (tid & 1) << 2,
        "load_b_row": tid >> 5,
        "load_b_col": (tid << 2) & 127,
        "c_row": ((warp_id >> 1 << 2) + (warp_lane & 3)) << 3,
        "c_col": (((warp_id & 1) << 3) + (warp_lane >> 2)) << 3,
    }


def pad(lines: list[str], width: int) -> list[str]:
    return [line.ljust(width) for line in lines]


def side_by_side(left_title: str, left: list[str], right_title: str, right: list[str]) -> str:
    left_width = max(len(left_title), *(len(x) for x in left)) if left else len(left_title)
    right_width = max(len(right_title), *(len(x) for x in right)) if right else len(right_title)
    left = pad(left, left_width)
    right = pad(right, right_width)
    out = []
    out.append("+" + "-" * (left_width + 2) + "+  +" + "-" * (right_width + 2) + "+")
    out.append(f"| {left_title.ljust(left_width)} |  | {right_title.ljust(right_width)} |")
    out.append("+" + "-" * (left_width + 2) + "+  +" + "-" * (right_width + 2) + "+")
    total = max(len(left), len(right))
    for i in range(total):
        l = left[i] if i < len(left) else " " * left_width
        r = right[i] if i < len(right) else " " * right_width
        out.append(f"| {l} |  | {r} |")
    out.append("+" + "-" * (left_width + 2) + "+  +" + "-" * (right_width + 2) + "+")
    return "\n".join(out)


def partition_map(selected_tid: int, gemm: str) -> list[str]:
    lines = []
    lines.append("     " + " ".join(f"c{c:02d}" for c in range(BLOCK)))
    lines.append("     " + "-----" * BLOCK)
    for pr in range(BLOCK):
        cells = []
        for pc in range(BLOCK):
            owner = None
            for tid in range(BLOCK * BLOCK):
                tx = tid % BLOCK
                ty = tid // BLOCK
                m = g1_mapping(tx, ty) if gemm == "g1" else g2_mapping(tx, ty)
                if m["c_row"] // TM == pr and m["c_col"] // TN == pc:
                    owner = tid
                    break
            if owner == selected_tid:
                cells.append(f"[{owner:03d}]")
            else:
                cells.append(f" {owner:03d} ")
        lines.append(f"r{pr:02d} |" + "|".join(cells) + "|")
    return lines


def small_summary(title: str, m: dict[str, int]) -> list[str]:
    lines = [
        f"threadIdx   = ({m['tx']}, {m['ty']})",
        f"tid         = {m['tid']}",
        f"A load      = row {m['load_a_row']}, col {m['load_a_col']}:{m['load_a_col'] + 4}",
        f"B load      = row {m['load_b_row']}, col {m['load_b_col']}:{m['load_b_col'] + 4}",
        f"C patch     = ({m['c_row']}:{m['c_row'] + 8}, {m['c_col']}:{m['c_col'] + 8})",
    ]
    if "warp_id" in m:
        lines.insert(2, f"warp/lane   = warp {m['warp_id']}, lane {m['warp_lane']}")
    return lines


def compare_same_thread(tx: int, ty: int) -> str:
    g1 = g1_mapping(tx, ty)
    g2 = g2_mapping(tx, ty)
    blocks = []
    blocks.append(side_by_side("GEMM_1 summary", small_summary("g1", g1), "GEMM_2 summary", small_summary("g2", g2)))
    blocks.append(side_by_side("GEMM_1 tile ownership", partition_map(g1["tid"], "g1"), "GEMM_2 tile ownership", partition_map(g2["tid"], "g2")))
    explanation = [
        "Key difference:",
        f"- GEMM_1 maps thread ({tx},{ty}) to a regular output patch at ({g1['c_row']},{g1['c_col']}).",
        f"- GEMM_2 maps the same thread to a warp-rearranged output patch at ({g2['c_row']},{g2['c_col']}).",
        "- A-load indexing is similar, but C-tile ownership is not.",
        "- This is one of the major readability/performance tradeoffs between the two kernels.",
    ]
    blocks.append("\n".join(explanation))
    return "\n\n".join(blocks)


def compare_same_patch(row_patch: int, col_patch: int) -> str:
    target_row = row_patch * TM
    target_col = col_patch * TN
    g1_match = None
    g2_match = None
    for tid in range(BLOCK * BLOCK):
        tx = tid % BLOCK
        ty = tid // BLOCK
        a = g1_mapping(tx, ty)
        b = g2_mapping(tx, ty)
        if a["c_row"] == target_row and a["c_col"] == target_col:
            g1_match = a
        if b["c_row"] == target_row and b["c_col"] == target_col:
            g2_match = b
    if g1_match is None or g2_match is None:
        raise SystemExit("Could not find matching patch in one of the kernels.")
    blocks = []
    blocks.append(side_by_side("GEMM_1 owner of patch", small_summary("g1", g1_match), "GEMM_2 owner of patch", small_summary("g2", g2_match)))
    blocks.append(side_by_side("GEMM_1 tile ownership", partition_map(g1_match["tid"], "g1"), "GEMM_2 tile ownership", partition_map(g2_match["tid"], "g2")))
    blocks.append(
        "\n".join(
            [
                f"Comparing the same output patch ({target_row}:{target_row + 8}, {target_col}:{target_col + 8}) across kernels:",
                f"- GEMM_1 owner threadIdx = ({g1_match['tx']},{g1_match['ty']}), tid = {g1_match['tid']}",
                f"- GEMM_2 owner threadIdx = ({g2_match['tx']},{g2_match['ty']}), tid = {g2_match['tid']}",
                "- GEMM_1 assigns patches in a straightforward row-major block layout.",
                "- GEMM_2 reorders ownership at warp granularity.",
            ]
        )
    )
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASCII comparison between gemm_1 and gemm_2")
    parser.add_argument("--thread-x", type=int)
    parser.add_argument("--thread-y", type=int)
    parser.add_argument("--patch-row", type=int)
    parser.add_argument("--patch-col", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thread_mode = args.thread_x is not None or args.thread_y is not None
    patch_mode = args.patch_row is not None or args.patch_col is not None
    if thread_mode and patch_mode:
        raise SystemExit("Use either thread mode or patch mode, not both.")
    if not thread_mode and not patch_mode:
        args.thread_x = 5
        args.thread_y = 2
        thread_mode = True
    if thread_mode:
        if args.thread_x is None or args.thread_y is None:
            raise SystemExit("thread mode requires both --thread-x and --thread-y.")
        if not (0 <= args.thread_x < BLOCK and 0 <= args.thread_y < BLOCK):
            raise SystemExit("thread-x and thread-y must be in [0,15].")
        print(compare_same_thread(args.thread_x, args.thread_y))
        return
    if args.patch_row is None or args.patch_col is None:
        raise SystemExit("patch mode requires both --patch-row and --patch-col.")
    if not (0 <= args.patch_row < BLOCK and 0 <= args.patch_col < BLOCK):
        raise SystemExit("patch-row and patch-col must be in [0,15].")
    print(compare_same_patch(args.patch_row, args.patch_col))


if __name__ == "__main__":
    main()
