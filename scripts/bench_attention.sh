#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13}"
export PATH="${PATH}:${CUDA_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64"

BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"
RUNNER="${BUILD_DIR}/attention_runner"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/attention_bench_${TIMESTAMP}.csv}"

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
VERIFY="${VERIFY:-0}"
CASES="${CASES:-1x8x128x64 1x8x256x64 1x8x512x64 1x16x256x128}"

mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[bench] attention runner missing, building first..."
  make -C "${ROOT_DIR}" attention
fi

mapfile -t kernels < <("${RUNNER}" --list-kernels)
if [[ ${#kernels[@]} -eq 0 ]]; then
  echo "[bench] no attention kernels registered" >&2
  exit 1
fi

printf 'kernel,batch,heads,seq_len,head_dim,warmup,iters,avg_ms,tflops,checksum,check_status,max_abs_diff\n' > "${OUT_CSV}"

echo "[bench] kernels: ${kernels[*]}"
echo "[bench] cases: ${CASES}"
echo "[bench] writing CSV: ${OUT_CSV}"

for shape in ${CASES}; do
  IFS='x' read -r batch heads seq_len head_dim <<< "${shape}"
  echo "[bench] case B=${batch} H=${heads} N=${seq_len} D=${head_dim}"

  for kernel in "${kernels[@]}"; do
    args=(
      --kernel "${kernel}"
      --batch "${batch}"
      --heads "${heads}"
      --seq-len "${seq_len}"
      --head-dim "${head_dim}"
      --warmup "${WARMUP}"
      --iters "${ITERS}"
    )
    if [[ "${VERIFY}" == "1" ]]; then
      args+=(--check)
    fi

    output="$(${RUNNER} "${args[@]}")"
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

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${kernel}" "${batch}" "${heads}" "${seq_len}" "${head_dim}" \
      "${WARMUP}" "${ITERS}" "${avg_ms}" "${tflops}" "${checksum}" \
      "${check_status}" "${max_abs_diff}" >> "${OUT_CSV}"
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
    key = (row["batch"], row["heads"], row["seq_len"], row["head_dim"])
    by_shape.setdefault(key, []).append(row)

for (batch, heads, seq_len, head_dim), group in by_shape.items():
    group.sort(key=lambda item: float(item["avg_ms"]))
    best = group[0]
    print(
        f"  B={batch} H={heads} N={seq_len} D={head_dim}: best={best['kernel']} avg_ms={float(best['avg_ms']):.4f} tflops={float(best['tflops']):.4f}"
    )
PY
