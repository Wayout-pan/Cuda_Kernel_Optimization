#!/usr/bin/env python3
"""Generate benchmark charts from the latest CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Schema:
    perf_col: str
    perf_label: str
    case_cols: tuple[str, ...]
    case_label: str


SCHEMAS = {
    "gemm": Schema("tflops", "TFLOPS", ("m", "n", "k"), "MxNxK"),
    "reduction": Schema("gbps", "GB/s", ("n",), "N"),
    "transpose": Schema("gbps", "GB/s", ("m", "n"), "MxN"),
    "attention": Schema("tflops", "TFLOPS", ("batch", "heads", "seq_len", "head_dim"), "BxHxNxD"),
}

BENCH_COMMANDS = {
    "gemm": "make bench-gemm",
    "reduction": "make bench-reduction",
    "transpose": "make bench-transpose",
    "attention": "make bench-attention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark charts from CSV files")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing *_bench_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to write PNG charts and an HTML index",
    )
    parser.add_argument(
        "--operators",
        nargs="*",
        default=sorted(SCHEMAS),
        choices=sorted(SCHEMAS),
        help="Operators to plot",
    )
    parser.add_argument(
        "--no-auto-bench",
        action="store_false",
        dest="auto_bench",
        help="Do not auto-run the matching benchmark when CSV data is missing",
    )
    parser.set_defaults(auto_bench=True)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def latest_csv(results_dir: Path, operator: str) -> Path | None:
    matches = sorted(results_dir.glob(f"{operator}_bench_*.csv"))
    return matches[-1] if matches else None


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def unique_in_order(items: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    ordered: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def case_to_label(case: tuple[str, ...]) -> str:
    return "x".join(case)


def pick_baseline(kernels: list[str]) -> str | None:
    for kernel in kernels:
        if kernel.endswith("_0"):
            return kernel
    return kernels[0] if kernels else None


def collect_case_labels(rows: list[dict[str, str]], schema: Schema) -> list[str]:
    cases = unique_in_order(tuple(row[col] for col in schema.case_cols) for row in rows)
    return [case_to_label(case) for case in cases]


def run_benchmark(operator: str) -> tuple[bool, str]:
    command = BENCH_COMMANDS[operator]
    print(f"[plot] {operator}: missing benchmark data, running `{command}`")
    completed = subprocess.run(command, cwd=repo_root(), shell=True)
    if completed.returncode != 0:
        return False, f"auto benchmark failed with exit code {completed.returncode}: `{command}`"
    return True, command


def ensure_rows(
    operator: str, schema: Schema, results_dir: Path, auto_bench: bool
) -> tuple[Path | None, list[dict[str, str]], str | None]:
    csv_path = latest_csv(results_dir, operator)
    rows = load_rows(csv_path) if csv_path is not None else []

    if (csv_path is None or not rows) and auto_bench:
        success, detail = run_benchmark(operator)
        if not success:
            return None, [], detail
        csv_path = latest_csv(results_dir, operator)
        rows = load_rows(csv_path) if csv_path is not None else []

    if csv_path is None:
        return None, [], f"no CSV found under {results_dir}. Run `{BENCH_COMMANDS[operator]}` first."

    if not rows:
        return (
            csv_path,
            [],
            f"CSV has no benchmark rows: {csv_path}. Re-run `{BENCH_COMMANDS[operator]}` to regenerate data.",
        )

    case_labels = collect_case_labels(rows, schema)
    if not case_labels:
        return csv_path, [], f"CSV has no valid shape columns: {csv_path}"

    return csv_path, rows, None


def plot_operator(
    operator: str, schema: Schema, rows: list[dict[str, str]], output_dir: Path
) -> tuple[Path, list[str]]:
    cases = unique_in_order(tuple(row[col] for col in schema.case_cols) for row in rows)
    case_labels = collect_case_labels(rows, schema)
    kernels = unique_in_order((row["kernel"],) for row in rows)
    kernel_names = [kernel[0] for kernel in kernels]
    perf_by_key = {
        (row["kernel"], tuple(row[col] for col in schema.case_cols)): float(row[schema.perf_col])
        for row in rows
    }
    latency_by_key = {
        (row["kernel"], tuple(row[col] for col in schema.case_cols)): float(row["avg_ms"])
        for row in rows
    }

    baseline = pick_baseline(kernel_names)
    shape_text = "Shapes: " + ", ".join(case_labels)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"{operator.upper()} benchmark overview\n{textwrap.fill(shape_text, width=90)}",
        fontsize=14,
    )
    positions = list(range(len(cases)))
    colors = ["#0f766e", "#1d4ed8", "#b45309", "#c026d3", "#dc2626", "#4338ca"]
    markers = ["o", "s", "^", "D", "P", "X"]

    for idx, kernel in enumerate(kernel_names):
        style = {
            "color": colors[idx % len(colors)],
            "marker": markers[idx % len(markers)],
            "linewidth": 2.2,
            "markersize": 6,
        }
        perf_values = [perf_by_key[(kernel, case)] for case in cases]
        latency_values = [latency_by_key[(kernel, case)] for case in cases]
        axes[0].plot(positions, perf_values, label=kernel, **style)
        axes[2].plot(positions, latency_values, label=kernel, **style)

        if baseline is not None:
            speedups = []
            for case in cases:
                baseline_perf = perf_by_key[(baseline, case)]
                current_perf = perf_by_key[(kernel, case)]
                speedups.append(current_perf / baseline_perf if baseline_perf else 0.0)
            axes[1].plot(positions, speedups, label=kernel, **style)

    axes[0].set_title(f"{operator.upper()} absolute performance trend")
    axes[0].set_ylabel(schema.perf_label)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].set_title(f"{operator.upper()} speedup trend vs {baseline}")
    axes[1].set_ylabel("x baseline")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle=":")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    axes[2].set_title(f"{operator.upper()} latency trend")
    axes[2].set_ylabel("avg_ms")
    axes[2].set_xlabel(schema.case_label)
    axes[2].set_xticks(positions, case_labels)
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{operator}_performance.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, case_labels


def write_index(output_dir: Path, generated: list[tuple[str, Path, Path, list[str]]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>CUDA Kernel Benchmark Charts</title>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 24px; background: #f5f7fb; color: #1b1f24; }",
        "    section { margin-bottom: 32px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }",
        "    img { max-width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 8px; }",
        "    code { background: #eef2ff; padding: 2px 6px; border-radius: 6px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>CUDA Kernel Benchmark Charts</h1>",
        "  <p>Each chart shows line trends for absolute performance, speedup versus baseline, and average latency.</p>",
    ]

    for operator, csv_path, image_path, case_labels in generated:
        lines.extend(
            [
                "  <section>",
                f"    <h2>{operator.upper()}</h2>",
                f"    <p>Source CSV: <code>{csv_path}</code></p>",
                f"    <p>Shapes: <code>{', '.join(case_labels)}</code></p>",
                f"    <img src='{image_path.name}' alt='{operator} benchmark chart'>",
                "  </section>",
            ]
        )

    lines.extend(["</body>", "</html>"])
    (output_dir / "index.html").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    generated: list[tuple[str, Path, Path, list[str]]] = []
    skipped: list[tuple[str, str]] = []

    for operator in args.operators:
        csv_path, rows, reason = ensure_rows(
            operator, SCHEMAS[operator], args.results_dir, args.auto_bench
        )
        if reason is not None:
            print(f"[plot] skip {operator}: {reason}")
            skipped.append((operator, reason))
            continue

        image_path, case_labels = plot_operator(operator, SCHEMAS[operator], rows, args.output_dir)
        generated.append((operator, csv_path, image_path, case_labels))
        print(f"[plot] wrote {image_path} shapes={', '.join(case_labels)}")

    if generated:
        write_index(args.output_dir, generated)
        print(f"[plot] wrote {args.output_dir / 'index.html'}")
    else:
        print("[plot] nothing generated")

    if skipped:
        print("[plot] missing benchmark data summary")
        for operator, reason in skipped:
            print(f"  - {operator}: {reason}")


if __name__ == "__main__":
    main()
