# Repository Guidelines

## Project Structure & Module Organization
- Core categories are: `sorts/`, `searches/`, `maths/`, `data_structures/`, `dynamic_programming/`, `graphs/`, `bit_manipulation/`, `conversions/`, `strings/`, `greedy_methods/`, `matrix/`, `backtracking/`.
- Each algorithm lives in one `.zig` file with both implementation and `test` blocks.
- [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig) is the test registry; add new algorithm files to `test_files` so they run in CI-style local checks.

## Build, Test, and Development Commands
- `zig version`: verify Zig toolchain (project is tested on `0.15.2`).
- `zig build test`: run all registered algorithm tests via [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig).
- `zig test sorts/bubble_sort.zig`: run tests for a single file while iterating.
- `zig fmt $(rg --files -g '*.zig')`: format all Zig sources.

## Coding Style & Naming Conventions
- Follow `zig fmt` output (do not hand-tune formatting).
- Use snake_case for file names (example: `cocktail_shaker_sort.zig`).
- Use lowerCamelCase for function names (example: `binarySearch`, `mergeSort`).
- Start files with:
  - module header comment (`//! <Algorithm> - Zig implementation`)
  - Python reference link (`//! Reference: ...`)
  - doc comments including time/space complexity.
- Prefer `comptime T: type` for generic numeric algorithms and make allocator ownership explicit for returned buffers.

## Testing Guidelines
- Keep tests in the same file as the algorithm.
- Name tests with pattern `test "algorithm: scenario"` (example: `test "binary search: not found"`).
- Cover normal, edge, and boundary cases (empty input, single item, sorted/reversed, negative values when applicable).
- For allocator-returning functions, free allocated memory in tests.

## Commit & Pull Request Guidelines
- Follow observed Conventional Commit style: `feat: ...`, `docs: ...`.
- Keep subjects concise and imperative; add a short body when listing added algorithms or test totals.
- Before opening a PR, confirm:
  - `zig build test` passes
  - new files are registered in [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig)
  - reference link and complexity comments are present in each new algorithm file.
- PR descriptions should include scope, touched directories, and local test evidence.

## Maintainer Decisions (Persistent)
- Review baseline is: algorithm correctness, boundary safety (no avoidable runtime panic), `zig build test` green, and documentation/metrics that reflect reality.
- Do **not** require strict 1:1 Python API cloning. Zig-idiomatic differences are acceptable when algorithmic outcome is equivalent and behavior differences are documented (e.g., `null`/error unions, output formatting, traversal order representation).
- Plan deviations are allowed, but must be explicitly recorded in both `README.md` and `EXPERIMENT_LOG.md` with rationale.
- Approved Phase 4 substitutions: `manacher -> is_pangram`, `roman_to_integer -> binary_to_hexadecimal` (accepted as schedule/scope tradeoff).
- Treat undocumented scope changes, contradictory batch stats, or mismatched summary numbers as defects that must be fixed before sync/release.

## Performance Benchmark Rules (Mandatory)
- Every newly added Zig algorithm that has a Python reference module must include a Python-vs-Zig performance comparison before merge.
- For each new algorithm, keep workload parity:
  - use equivalent input generation strategy in Python and Zig
  - use equivalent iteration count and warm-up behavior
  - include checksum/output guard to prevent dead-code elimination and to validate comparable results
- During incremental development, single-algorithm comparison is required first:
  - add/update a focused benchmark case for that algorithm
  - record the result into `benchmarks/python_vs_zig` artifacts
- For benchmark artifact maintenance after adding/updating an algorithm:
  - update benchmark harness inputs/cases in `benchmarks/python_vs_zig/python_bench_all.py` and `benchmarks/python_vs_zig/zig_bench_all.zig`
  - regenerate outputs via `bash benchmarks/python_vs_zig/run_all.sh` (or equivalent targeted run + `python3 benchmarks/python_vs_zig/build_leaderboard.py`)
  - keep `leaderboard_all.md`, `leaderboard_all.csv`, `category_speedup_chart.csv`, and `summary_all.md` in sync
- If an algorithm cannot be fairly aligned with a single-call Python counterpart (for example stateful data structures or traversal-order-sensitive backtracking), document the reason and the chosen benchmark protocol in `README.md` and `EXPERIMENT_LOG.md` before merge.
