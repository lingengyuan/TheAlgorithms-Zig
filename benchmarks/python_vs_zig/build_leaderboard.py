#!/usr/bin/env python3
"""Merge Python/Zig benchmark CSV outputs into leaderboard and chart data."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> None:
    root = Path(__file__).resolve().parent
    py_path = root / "python_results_all.csv"
    zig_path = root / "zig_results_all.csv"

    py_rows = {r["algorithm"]: r for r in csv.DictReader(py_path.open())}
    zig_rows = {r["algorithm"]: r for r in csv.DictReader(zig_path.open())}

    missing_py = sorted(set(zig_rows) - set(py_rows))
    missing_zig = sorted(set(py_rows) - set(zig_rows))
    if missing_py or missing_zig:
        raise SystemExit(f"mismatched algorithms: missing_py={missing_py}, missing_zig={missing_zig}")

    merged = []
    for alg, py in py_rows.items():
        zig = zig_rows[alg]
        py_total = int(py["total_ns"])
        zig_total = int(zig["total_ns"])
        py_iter = int(py["iterations"])
        zig_iter = int(zig["iterations"])
        py_avg = py_total / py_iter
        zig_avg = zig_total / zig_iter
        speedup = py_avg / zig_avg if zig_avg > 0 else float("inf")
        merged.append(
            {
                "algorithm": alg,
                "category": py["category"],
                "python_avg_ns": py_avg,
                "zig_avg_ns": zig_avg,
                "speedup_py_over_zig": speedup,
                "checksum_match": py["checksum"] == zig["checksum"],
            }
        )

    merged.sort(key=lambda r: r["speedup_py_over_zig"], reverse=True)

    leaderboard_csv = root / "leaderboard_all.csv"
    with leaderboard_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "category", "python_avg_ns", "zig_avg_ns", "speedup_py_over_zig", "checksum_match"])
        for r in merged:
            w.writerow(
                [
                    r["algorithm"],
                    r["category"],
                    f"{r['python_avg_ns']:.2f}",
                    f"{r['zig_avg_ns']:.2f}",
                    f"{r['speedup_py_over_zig']:.2f}",
                    str(r["checksum_match"]).lower(),
                ]
            )

    category_to_speedups: dict[str, list[float]] = {}
    for r in merged:
        category_to_speedups.setdefault(r["category"], []).append(r["speedup_py_over_zig"])

    chart_csv = root / "category_speedup_chart.csv"
    with chart_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "algorithm_count", "mean_speedup", "median_speedup"])
        for category in sorted(category_to_speedups):
            speeds = category_to_speedups[category]
            w.writerow(
                [
                    category,
                    len(speeds),
                    f"{(sum(speeds) / len(speeds)):.2f}",
                    f"{median(speeds):.2f}",
                ]
            )

    leaderboard_md = root / "leaderboard_all.md"
    lines = [
        f"# Python vs Zig Benchmark Leaderboard ({len(merged)} alignable algorithms)",
        "",
        "| Rank | Algorithm | Category | Python avg ns | Zig avg ns | Speedup (Python/Zig) | Checksum |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for i, r in enumerate(merged, 1):
        lines.append(
            f"| {i} | {r['algorithm']} | {r['category']} | {r['python_avg_ns']:.2f} | "
            f"{r['zig_avg_ns']:.2f} | {r['speedup_py_over_zig']:.2f}x | {str(r['checksum_match']).lower()} |"
        )
    leaderboard_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    speedups = [r["speedup_py_over_zig"] for r in merged]
    mean_speedup = sum(speedups) / len(speedups)
    median_speedup = median(speedups)
    geometric_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    checksum_matches = sum(1 for r in merged if r["checksum_match"])

    summary_md = root / "summary_all.md"
    summary_lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Alignable algorithms benchmarked | {len(merged)} |",
        f"| Checksum match count | {checksum_matches} |",
        f"| Mean speedup (Python/Zig) | {mean_speedup:.2f}x |",
        f"| Median speedup (Python/Zig) | {median_speedup:.2f}x |",
        f"| Geometric mean speedup (Python/Zig) | {geometric_mean:.2f}x |",
    ]
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
