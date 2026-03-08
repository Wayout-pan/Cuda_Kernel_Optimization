#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"
RUNNER="${BUILD_DIR}/gemm_runner"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/gemm_bench_${TIMESTAMP}.csv}"

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
VERIFY="${VERIFY:-0}"
CASES="${CASES:-1024x1024x1024 2048x2048x2048 4096x4096x4096}"

mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[bench] runner missing, building first..."
  make -C "${ROOT_DIR}" all
fi

mapfile -t kernels < <("${RUNNER}" --list-kernels)
if [[ ${#kernels[@]} -eq 0 ]]; then
  echo "[bench] no kernels registered" >&2
  exit 1
fi

printf 'kernel,m,n,k,warmup,iters,avg_ms,tflops,checksum,check_status,max_abs_diff\n' > "${OUT_CSV}"

echo "[bench] kernels: ${kernels[*]}"
echo "[bench] cases: ${CASES}"
echo "[bench] writing CSV: ${OUT_CSV}"

for shape in ${CASES}; do
  IFS='x' read -r m n k <<< "${shape}"
  echo "[bench] case M=${m} N=${n} K=${k}"

  for kernel in "${kernels[@]}"; do
    args=(--kernel "${kernel}" --m "${m}" --n "${n}" --k "${k}" --warmup "${WARMUP}" --iters "${ITERS}")
    if [[ "${VERIFY}" == "1" ]]; then
      args+=(--check)
    fi

    output="$("${RUNNER}" "${args[@]}")"
    echo "${output}"

    result_line="$(printf '%s\n' "${output}" | awk '/^RESULT / {print; exit}')"
    check_line="$(printf '%s\n' "${output}" | awk '/^CHECK / {print; exit}')"

    avg_ms="$(printf '%s\n' "${result_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^avg_ms=/){split($i,a,"="); print a[2]}}}')"
    tflops="$(printf '%s\n' "${result_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^tflops=/){split($i,a,"="); print a[2]}}}')"
    checksum="$(printf '%s\n' "${result_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^checksum=/){split($i,a,"="); print a[2]}}}')"

    check_status="na"
    max_abs_diff="na"
    if [[ -n "${check_line}" ]]; then
      check_status="$(printf '%s\n' "${check_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^skipped=/){split($i,a,"="); print a[2]}}}')"
      max_abs_diff="$(printf '%s\n' "${check_line}" | awk '{for(i=1;i<=NF;i++){if($i ~ /^max_abs_diff=/){split($i,a,"="); print a[2]}}}')"
      [[ -z "${max_abs_diff}" ]] && max_abs_diff="na"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${kernel}" "${m}" "${n}" "${k}" "${WARMUP}" "${ITERS}" \
      "${avg_ms}" "${tflops}" "${checksum}" "${check_status}" "${max_abs_diff}" \
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

print("[bench] summary by shape")
by_shape = {}
for row in rows:
    key = (row["m"], row["n"], row["k"])
    by_shape.setdefault(key, []).append(row)

for (m, n, k), group in by_shape.items():
    group.sort(key=lambda item: float(item["avg_ms"]))
    best = group[0]
    print(f"  M={m} N={n} K={k}: best={best['kernel']} avg_ms={float(best['avg_ms']):.4f} tflops={float(best['tflops']):.4f}")
PY
