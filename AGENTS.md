# Repository Guidelines

## Project Structure & Module Organization
- `sorts/`, `searches/`, and `maths/` contain implemented algorithms.
- `data_structures/` and `dynamic_programming/` are placeholders for upcoming work.
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
