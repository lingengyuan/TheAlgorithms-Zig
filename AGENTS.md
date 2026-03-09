# Repository Guidelines

## Project Structure & Module Organization
- Core categories are: `sorts/`, `searches/`, `maths/`, `data_structures/`, `dynamic_programming/`, `graphs/`, `bit_manipulation/`, `conversions/`, `strings/`, `greedy_methods/`, `matrix/`, `backtracking/`, `project_euler/`.
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
- Mandatory (Phase 4 rule #3): every algorithm test set must include extreme-case scenarios to verify correctness under stress/edge conditions (for example: empty/min/max inputs, degenerate structures, overflow-prone values, invalid/out-of-range inputs, and disconnected/unreachable cases when applicable).
- For allocator-returning functions, free allocated memory in tests.
- Test execution cadence (updated):
  - during implementation, run affected file-level tests with `zig test <file>` (mandatory)
  - run full `zig build test` at least once per every 2 waves (recommended)
  - run full `zig build test` before each commit/push (mandatory)

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
- Plan deviations are allowed, but must be explicitly recorded in both `README.md` and the relevant experiment-log entry referenced from `EXPERIMENT_LOG.md`, with rationale.
- Approved Phase 4 substitutions: `manacher -> is_pangram`, `roman_to_integer -> binary_to_hexadecimal` (accepted as schedule/scope tradeoff).
- Treat undocumented scope changes, contradictory batch stats, or mismatched summary numbers as defects that must be fixed before sync/release.

## Python Consistency Rules (Mandatory)
- Python reference project for behavior and expected results: `https://github.com/TheAlgorithms/Python`.
- Do not add Python-vs-Zig performance comparison code or records to this repository.
- For each Zig algorithm that has a Python reference module, prioritize functional consistency with the Python implementation (same algorithmic outcome under equivalent input domain).
- Test expectations must align with Python reference behavior. If Zig introduces intentional API differences, keep output semantics equivalent and document the difference clearly in code comments/tests.
- `EXPERIMENT_LOG.md` is the bilingual experiment-log index. When the log grows too large, keep the root file as a bilingual summary/index and split detailed records into linked date-based files under `docs/experiment_logs/phase5/by-date/`.
- All new experiment-log content must be bilingual (English + Simplified Chinese). When logs are split, append the entry to the correct date file while keeping the batch/wave section inside that dated log.
- Experimental integrity is mandatory: any real errors encountered during implementation/testing (compile errors, runtime panics, overflow/data-range issues, logic mismatches) must be truthfully logged in the relevant batch section inside the dated file referenced by `EXPERIMENT_LOG.md`.
  - each recorded issue must include: failing step/command, error symptom, root cause, fix applied, and post-fix verification result
