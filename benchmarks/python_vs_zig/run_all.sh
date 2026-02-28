#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Build Zig benchmark binary..."
ZIG_LOCAL_CACHE_DIR=.zig-cache ZIG_GLOBAL_CACHE_DIR=.zig-cache \
  zig build-exe benchmarks_python_vs_zig_all.zig -O ReleaseFast \
  -femit-bin=benchmarks/python_vs_zig/zig_bench_all

echo "[2/4] Run Zig benchmark..."
./benchmarks/python_vs_zig/zig_bench_all > benchmarks/python_vs_zig/zig_results_all.csv

echo "[3/4] Run Python benchmark..."
python3 benchmarks/python_vs_zig/python_bench_all.py > benchmarks/python_vs_zig/python_results_all.csv

echo "[4/4] Build leaderboard + chart data..."
python3 benchmarks/python_vs_zig/build_leaderboard.py

echo "Done."
echo "Outputs:"
echo "  - benchmarks/python_vs_zig/leaderboard_all.md"
echo "  - benchmarks/python_vs_zig/leaderboard_all.csv"
echo "  - benchmarks/python_vs_zig/category_speedup_chart.csv"
echo "  - benchmarks/python_vs_zig/summary_all.md"
