#!/usr/bin/env python3
"""Upsert single-algorithm benchmark rows into full benchmark CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not path.exists():
        return [], {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = {row["algorithm"]: row for row in reader}
    return fieldnames, rows


def write_rows(path: Path, fieldnames: list[str], rows: dict[str, dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alg in sorted(rows):
            writer.writerow(rows[alg])


def upsert(single_path: Path, full_path: Path) -> None:
    single_fields, single_rows = read_rows(single_path)
    if len(single_rows) != 1:
        raise SystemExit(f"expected exactly one row in {single_path}, got {len(single_rows)}")

    full_fields, full_rows = read_rows(full_path)
    if full_fields and single_fields and full_fields != single_fields:
        raise SystemExit(f"schema mismatch between {single_path} and {full_path}")

    fieldnames = full_fields or single_fields
    if not fieldnames:
        raise SystemExit(f"missing CSV header in {single_path}")

    for alg, row in single_rows.items():
        full_rows[alg] = row

    write_rows(full_path, fieldnames, full_rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    upsert(root / "python_results_single.csv", root / "python_results_all.csv")
    upsert(root / "zig_results_single.csv", root / "zig_results_all.csv")


if __name__ == "__main__":
    main()
