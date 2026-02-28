# TheAlgorithms-Zig — Project Instructions for Claude

## Project Overview

This is a Zig (0.15.2) implementation of classic algorithms, ported from
[TheAlgorithms/Python](https://github.com/TheAlgorithms/Python).
It is also a vibe coding experiment: AI translates Python → Zig, and every
batch is logged for success rates and failure patterns.

---

## Mandatory Steps After Adding Any Algorithm

Whenever one or more `.zig` algorithm files are added or changed, **always**
complete all of the following before committing, without waiting to be asked:

### 1. Register in `build.zig`

Add the new file path to the `test_files` array in `build.zig` under the
correct category comment. Run `zig build test` to confirm all tests pass.

### 2. Update `README.md` (bilingual)

- Add a row to the correct algorithm table in **both** the English section and
  the Chinese (简体中文) section.
- Update the section header count, e.g. `### Sorting (12)` → `(13)`.
- Update the project structure tree counts if a new directory was created.
- Format: `| Algorithm Name | [\`path/file.zig\`](path/file.zig) | O(...) |`

### 3. Update `EXPERIMENT_LOG.md` (bilingual)

Record the batch entry in **both** the English section and the Chinese section.
Each algorithm must include:

| Field | What to record |
|-------|---------------|
| Compile attempts | How many generations before it compiled |
| Test pass | `✅ N/N` if all passed first try, `❌ X/N failed` with failure details if not |
| Manual fix lines | Exact number of lines changed by human after AI generation |
| Notes | Zig-specific patterns, API surprises, edge cases handled |

**Accuracy rules — do not sanitise failures:**
- If a test assertion was wrong (even if the algorithm logic is correct), record
  it as a failure with the label `test assertion error`.
- If a compile error occurred, record the error type and what caused it.
- If a runtime panic was found during review, record it as a QA finding.
- Distinguish between: compile fail / test assertion fail / runtime safety fail.

After each batch, update the **Cumulative Summary** table in both languages.

---

## Commit Convention

```
feat: add <batch description> — N new algorithms, M tests

<category>: <algo1>, <algo2>, ...

Total project: X algorithms, Y tests, all green.
<any notable failures or fixes>

Co-Authored-By: Claude <model> <noreply@anthropic.com>
```

---

## Code Conventions

- Each `.zig` file must start with:
  ```zig
  //! <Algorithm Name> - Zig implementation
  //! Reference: https://github.com/TheAlgorithms/Python/blob/master/<path>.py
  ```
  For algorithms with no direct Python equivalent (e.g. BFS/DFS written from
  scratch), use a Wikipedia reference instead.

- Every file is self-contained: implementation + `test` blocks in one file.
- Use `comptime T: type` generics where the algorithm is type-agnostic.
- Use `testing.allocator` in tests for any heap allocation (detects leaks).
- `defer allocator.free(...)` for all allocations within a function scope.

## Zig 0.15 API Notes (AI tends to get these wrong)

- `build.zig`: use `root_module = b.createModule(.{ .root_source_file = ..., .target = ..., .optimize = ... })` inside `addTest`.
- `ArrayListUnmanaged.pop()` returns `?T`, not `T` — must unwrap with `.?` or `orelse`.
- `ArrayListUnmanaged` methods all take `allocator` as first argument.
- No `&&` or `||` — use `and` / `or`.
- Unsigned integer subtraction underflows in debug mode — add guards before subtracting.

## Directory Structure

```
TheAlgorithms-Zig/
├── build.zig
├── build.zig.zon
├── sorts/
├── searches/
├── maths/
├── data_structures/
├── dynamic_programming/
├── graphs/
├── bit_manipulation/
├── conversions/
├── strings/
├── backtracking/        # WIP
├── matrix/              # WIP
└── greedy_methods/      # WIP
```

## GitHub

Remote: `https://github.com/lingengyuan/TheAlgorithms-Zig`
Branch: `main`
Push using `GH_TOKEN` environment variable (user provides when needed).
