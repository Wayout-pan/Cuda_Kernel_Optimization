#!/usr/bin/env bash
set -euo pipefail

# 这个脚本负责批量 benchmark。
# 它会：
# 1. 确保 runner 已经编译好
# 2. 自动获取所有已注册 kernel
# 3. 对多组 MxNxK 逐个跑分
# 4. 把结果写到 CSV
# 5. 最后输出每个 shape 的最快 kernel

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"
RUNNER="${BUILD_DIR}/gemm_runner"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/gemm_bench_${TIMESTAMP}.csv}"

# 这些环境变量都可以在命令行临时覆盖。
# 例如：WARMUP=5 ITERS=20 VERIFY=1 bash scripts/bench_gemm.sh
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
VERIFY="${VERIFY:-0}"
CASES="${CASES:-1024x1024x1024 2048x2048x2048 4096x4096x4096}"

mkdir -p "${RESULTS_DIR}"

# 如果 runner 不存在，就先编译。
if [[ ! -x "${RUNNER}" ]]; then
  echo "[bench] runner missing, building first..."
  make -C "${ROOT_DIR}" all
fi

# 通过 runner 的 --list-kernels 自动发现当前有哪些 kernel。
# 这样新增 kernel 后，脚本通常不需要改。
mapfile -t kernels < <("${RUNNER}" --list-kernels)
if [[ ${#kernels[@]} -eq 0 ]]; then
  echo "[bench] no kernels registered" >&2
  exit 1
fi

# 先写 CSV 表头。
printf 'kernel,m,n,k,warmup,iters,avg_ms,tflops,checksum,check_status,max_abs_diff\n' > "${OUT_CSV}"

echo "[bench] kernels: ${kernels[*]}"
echo "[bench] cases: ${CASES}"
echo "[bench] writing CSV: ${OUT_CSV}"

# 外层循环遍历不同测试尺寸。
for shape in ${CASES}; do
  # 把形如 1024x1024x1024 的字符串拆成 m/n/k。
  IFS='x' read -r m n k <<< "${shape}"
  echo "[bench] case M=${m} N=${n} K=${k}"

  # 内层循环遍历每个 kernel。
  for kernel in "${kernels[@]}"; do
    args=(--kernel "${kernel}" --m "${m}" --n "${n}" --k "${k}" --warmup "${WARMUP}" --iters "${ITERS}")
    if [[ "${VERIFY}" == "1" ]]; then
      args+=(--check)
    fi

    # 执行 runner，并把完整输出保存下来，便于后面解析。
    output="$(${RUNNER} "${args[@]}")"
    echo "${output}"

    # RESULT 行包含时间、TFLOPS、checksum。
    # CHECK 行包含是否跳过校验以及最大误差。
    result_line="$(printf '%s\n' "${output}" | awk '/^RESULT / {print; exit}')"
    check_line="$(printf '%s\n' "${output}" | awk '/^CHECK / {print; exit}')"

    # 从 RESULT 行里抽取字段。
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

    # 将本次运行结果追加进 CSV。
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${kernel}" "${m}" "${n}" "${k}" "${WARMUP}" "${ITERS}" \
      "${avg_ms}" "${tflops}" "${checksum}" "${check_status}" "${max_abs_diff}" \
      >> "${OUT_CSV}"
  done

done

# 用一个很短的 Python 片段读取 CSV，打印每个 shape 的最快 kernel。
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
