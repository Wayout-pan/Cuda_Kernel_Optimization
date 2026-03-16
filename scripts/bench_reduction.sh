#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"
RUNNER="${BUILD_DIR}/reduction_runner"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/reduction_bench_${TIMESTAMP}.csv}"

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
VERIFY="${VERIFY:-0}"
CASES="${CASES:-1048576 4194304 16777216 67108864}"

mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[bench] reduction runner missing, building first..."
  make -C "${ROOT_DIR}" reduction
fi

mapfile -t kernels < <("${RUNNER}" --list-kernels)
if [[ ${#kernels[@]} -eq 0 ]]; then
  echo "[bench] no reduction kernels registered" >&2
  exit 1
fi

printf 'kernel,n,warmup,iters,avg_ms,gbps,value,check_status,abs_diff,rel_diff\n' > "${OUT_CSV}"

echo "[bench] kernels: ${kernels[*]}"
echo "[bench] cases: ${CASES}"
echo "[bench] writing CSV: ${OUT_CSV}"

for n in ${CASES}; do
  echo "[bench] case N=${n}"

  for kernel in "${kernels[@]}"; do
    args=(--kernel "${kernel}" --n "${n}" --warmup "${WARMUP}" --iters "${ITERS}")
    if [[ "${VERIFY}" == "1" ]]; then
      args+=(--check)
    fi

    output="$(${RUNNER} "${args[@]}")"
    echo "${output}"

    result_line="$(printf '%s\n' "${output}" | awk '/^RESULT / {print; exit}')"
    check_line="$(printf '%s\n' "${output}" | awk '/^CHECK / {print; exit}')"

    avg_ms="$(printf '%s\n' "${result_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^avg_ms=/){split($i,a,"="); print a[2]}}}')"
    gbps="$(printf '%s\n' "${result_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^gbps=/){split($i,a,"="); print a[2]}}}')"
    value="$(printf '%s\n' "${result_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^value=/){split($i,a,"="); print a[2]}}}')"

    check_status="na"
    abs_diff="na"
    rel_diff="na"
    if [[ -n "${check_line}" ]]; then
      check_status="$(printf '%s\n' "${check_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^skipped=/){split($i,a,"="); print a[2]}}}')"
      abs_diff="$(printf '%s\n' "${check_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^abs_diff=/){split($i,a,"="); print a[2]}}}')"
      rel_diff="$(printf '%s\n' "${check_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^rel_diff=/){split($i,a,"="); print a[2]}}}')"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${kernel}" "${n}" "${WARMUP}" "${ITERS}" \
      "${avg_ms}" "${gbps}" "${value}" "${check_status}" "${abs_diff}" "${rel_diff}" \
      >> "${OUT_CSV}"
  done
done

python3 - <<'PY' "${OUT_CSV}"
import csv
import pathlib
import sys

csv_path = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open()))
if not rows:
    raise SystemExit(0)

print("[bench] summary by N")
by_n = {}
for row in rows:
    by_n.setdefault(row["n"], []).append(row)

for n, group in by_n.items():
    group.sort(key=lambda item: float(item["avg_ms"]))
    best = group[0]
    print(
        f"  N={n}: best={best['kernel']} avg_ms={float(best['avg_ms']):.4f} gbps={float(best['gbps']):.4f}"
    )
PY
