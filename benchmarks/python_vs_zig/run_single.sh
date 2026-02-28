#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <algorithm_name>"
  exit 1
fi

ALGORITHM="$1"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Build Zig benchmark binary..."
ZIG_LOCAL_CACHE_DIR=.zig-cache ZIG_GLOBAL_CACHE_DIR=.zig-cache \
  zig build-exe benchmarks_python_vs_zig_all.zig -O ReleaseFast \
  -femit-bin=benchmarks/python_vs_zig/zig_bench_all

echo "[2/5] Run Zig benchmark for '${ALGORITHM}'..."
BENCH_ALGORITHM="$ALGORITHM" ./benchmarks/python_vs_zig/zig_bench_all > benchmarks/python_vs_zig/zig_results_single.csv

echo "[3/5] Run Python benchmark for '${ALGORITHM}'..."
BENCH_ALGORITHM="$ALGORITHM" python3 benchmarks/python_vs_zig/python_bench_all.py > benchmarks/python_vs_zig/python_results_single.csv

echo "[4/5] Merge single-algorithm results into full CSVs..."
python3 benchmarks/python_vs_zig/merge_single_result.py

echo "[5/5] Rebuild leaderboard + chart data..."
python3 benchmarks/python_vs_zig/build_leaderboard.py

echo "Done."
echo "Updated:"
echo "  - benchmarks/python_vs_zig/python_results_all.csv"
echo "  - benchmarks/python_vs_zig/zig_results_all.csv"
echo "  - benchmarks/python_vs_zig/leaderboard_all.md"
echo "  - benchmarks/python_vs_zig/leaderboard_all.csv"
echo "  - benchmarks/python_vs_zig/category_speedup_chart.csv"
echo "  - benchmarks/python_vs_zig/summary_all.md"
