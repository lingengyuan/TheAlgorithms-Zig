# Vibe Coding Experiment Log

> AI-assisted translation of Python algorithms to Zig (0.15.2), recording success rates and failure patterns.

## Phase 1: Spike (5 algorithms)

**Date:** 2026-02-27
**Model:** Claude Opus 4.6
**Go/No-Go Result:** Go (5/5 first-attempt pass, threshold was 4/5)

---

### linear_search — 2026-02-27

- AI attempts to compile: **1** (first attempt passed)
- Error types: None
- Manual fix lines: **0**
- Test cases: 7, all passed
- Notes: Simplest algorithm. Used Zig's `for (items, 0..)` indexed iteration — idiomatic and clean.

### bubble_sort — 2026-02-27

- AI attempts to compile: **1** (first attempt passed)
- Error types: None
- Manual fix lines: **0**
- Test cases: 5, all passed
- Notes: Direct translation from Python. `while (i < n) : (i += 1)` maps naturally to Python's `for i in range(n)`.

### insertion_sort — 2026-02-27

- AI attempts to compile: **1** (first attempt passed)
- Error types: None
- Manual fix lines: **0**
- Test cases: 6, all passed
- Notes: Key Zig difference: `and` instead of `&&` for boolean logic. `usize` decrement in the inner loop requires care but no special handling since the `j > 0` guard runs first.

### binary_search — 2026-02-27

- AI attempts to compile: **1** (first attempt passed)
- Error types: None
- Manual fix lines: **0**
- Test cases: 9, all passed
- Notes: Returned `?usize` (optional) instead of Python's `-1` convention — more idiomatic Zig. Added `mid == 0` guard to prevent `usize` underflow on `high = mid - 1`.

### merge_sort — 2026-02-27

- AI attempts to compile: **1** (first attempt passed)
- Error types: None
- Manual fix lines: **0**
- Test cases: 7, all passed
- Notes: First algorithm requiring `std.mem.Allocator`. Used `testing.allocator` in tests (catches memory leaks automatically). `defer allocator.free()` pattern for RAII-style cleanup.

---

## Phase 2 Batch 1: Easy algorithms (15 algorithms)

**Date:** 2026-02-27
**Model:** Claude Opus 4.6
**Batch result:** 15/15 first-attempt pass (100%)

### Sorts (5 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| selection_sort | 1 | 0 | 5 | Used Zig 0.15 `for (0..n)` range syntax — cleaner than `while` loops |
| shell_sort | 1 | 0 | 6 | Ciura gap sequence as comptime array. `continue` to skip gaps larger than array |
| counting_sort | 1 | 0 | 5 | Non-generic (i32 only) — needs `@intCast` for index math. Uses allocator |
| cocktail_shaker_sort | 1 | 0 | 5 | Bidirectional bubble sort. Right-to-left pass uses `while (j > start)` to avoid usize underflow |
| gnome_sort | 1 | 0 | 5 | Simplest sort — ~15 lines. No range-for, just a while loop with manual index |

### Searches (2 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| jump_search | 1 | 0 | 6 | Used `math.sqrt` for block size and `@min` builtin for bounds |
| ternary_search | 1 | 0 | 7 | Falls back to linear scan when range < 3 to avoid usize division edge cases |

### Maths (8 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| gcd | 1 | 0 | 3 | Handles negative inputs via `@intCast` to u64. Returns u64 |
| lcm | 1 | 0 | 4 | `a / gcd(a,b) * b` to avoid overflow (divide before multiply) |
| fibonacci | 1 | 0 | 3 | Iterative, returns u64. Clean `for (2..n+1)` range |
| prime_check | 1 | 0 | 4 | 6k±1 optimization. `i * i <= n` avoids sqrt call |
| sieve_of_eratosthenes | 1 | 0 | 5 | Uses allocator for bool sieve + result array. `@memset` for init |
| power | 1 | 0 | 3 | Both `power(i64)` and `powerMod(u64)` variants |
| factorial | 1 | 0 | 3 | Iterative. `u64` holds up to 20! (2.4×10¹⁸) |
| collatz_sequence | 1 | 0 | 4 | Two APIs: `collatzSteps` (count only) and `collatzSequence` (buffer fill) |

---

## Phase 2 Batch 2A: Medium algorithms (#16-#21, 6 algorithms)

**Date:** 2026-02-27
**Model:** Codex GPT-5
**Batch result:** 6 implemented, 4/6 first-attempt compile pass

### Sorts (4 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| quick_sort | 1 | 0 | 6 | In-place Lomuto partition with recursive slice splitting |
| heap_sort | 1 | 0 | 6 | Max-heap + sift-down, fully in-place |
| radix_sort | 1 | 0 | 6 | Splits negatives/positives, runs radix on unsigned buffers |
| bucket_sort | 1 | 1 | 6 | Sqrt(n) buckets + per-bucket insertion sort; fixed signed division with `@divTrunc` |

### Searches (2 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| exponential_search | 1 | 0 | 6 | Exponential bound expansion + bounded binary search |
| interpolation_search | 1 | 1 | 6 | Probe-position formula; fixed signed division with `@divTrunc` |

---

## Phase 2 Batch 2B: Medium algorithms (#22-#28, 7 algorithms)

**Date:** 2026-02-27
**Model:** Codex GPT-5
**Batch result:** 7/7 first-attempt compile pass (100%)

### Data Structures (2 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| stack | 1 | 0 | 3 | Generic dynamic-array stack with amortized O(1) push/pop |
| queue | 1 | 0 | 3 | Generic circular-buffer queue with dynamic growth |

### Dynamic Programming (5 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| climbing_stairs | 1 | 0 | 3 | Classic 1D DP recurrence with O(1) space |
| fibonacci_dp | 1 | 0 | 3 | Top-down memoized Fibonacci (`allocator` + memo table) |
| coin_change | 1 | 0 | 5 | 1D DP for minimum coin count, returns `null` if impossible |
| max_subarray_sum | 1 | 0 | 5 | Kadane single-pass maximum subarray |
| longest_common_subsequence | 1 | 0 | 5 | 2D DP table (flattened allocation) returning LCS length |

---

## Phase 2 QA Hotfix: Safety + semantic alignment (no new algorithms)

**Date:** 2026-02-27
**Model:** Codex GPT-5
**Batch result:** all regressions fixed, `zig build test` fully green

### Issues found during review

- Runtime panics on edge input:
  - `gcd/lcm` overflow on `minInt(i64)`
  - `powerMod` divide-by-zero when `modulus == 0`
  - `counting_sort` overflow/huge-range allocation blow-up
  - `collatz_sequence` potential out-of-bounds write on small buffers
- Semantic mismatches against Python reference modules:
  - `coin_change`: minimum coins vs number-of-ways
  - `fibonacci_dp`: single value vs sequence output
  - `longest_common_subsequence`: length-only vs `(length, subsequence)`
  - `max_subarray_sum`: empty-input behavior / `allow_empty_subarrays` mode
  - `climbing_stairs`: missing positive-input validation
  - `stack/queue`: missing Python-style error-oriented APIs

### Fix summary

- Added explicit edge guards and error returns in affected algorithms.
- Aligned DP algorithm outputs with referenced Python behavior where required.
- Added Python-style error APIs (`StackOverflow/Underflow`, `EmptyQueue`) while keeping existing ergonomics.
- Added targeted regression tests for each discovered failure mode.
- Manual fixes applied: **305 insertions, 119 deletions** across 12 algorithm files.

---

## Phase 3: Hard algorithms (#29-#36, 8 algorithms)

**Date:** 2026-02-27
**Model:** Claude Opus 4.6
**Batch result:** 7/8 first-attempt compile pass (DFS needed 1-line fix)

### Data Structures (4 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| singly_linked_list | 1 | 0 | 6 | Generic via `pub fn SinglyLinkedList(comptime T: type) type`. `allocator.create/destroy` for nodes. `?*Node` pointer chains |
| doubly_linked_list | 1 | 0 | 7 | head+tail dual pointers. `insertTail` O(1). Reverse swaps `prev`/`next` on every node + swaps head/tail |
| binary_search_tree | 1 | 0 | 6 | Recursive insert/search/remove. In-order successor replacement for two-child delete. `deinit` recursively frees subtree |
| min_heap | 1 | 0 | 5 | `ArrayListUnmanaged` backing array. siftUp/siftDown. `fromSlice` heapify in O(n) |

### Dynamic Programming (2 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| edit_distance | 1 | 0 | 6 | Flattened 2D DP table `(m+1)×(n+1)`. Bottom-up with insert/delete/replace |
| knapsack | 1 | 0 | 6 | Flattened 2D DP table `(n+1)×(W+1)`. Standard 0/1 take-or-skip recurrence |

### Graphs (2 new)

| Algorithm | Attempts | Manual fixes | Tests | Notes |
|-----------|----------|-------------|-------|-------|
| bfs | 1 | 0 | 4 | ArrayList as queue + visited array. Adjacency list as `[]const []const usize` |
| dfs | **2** | **1** | 5 | `ArrayListUnmanaged.pop()` returns `?usize` in Zig 0.15 — AI forgot `.?` unwrap. 1-line fix |

---

## Phase 3 QA Hotfix: Boundary safety hardening

**Date:** 2026-02-27
**Model:** Claude Opus 4.6
**Batch result:** 4 issues fixed, 3 regression tests added, `zig build test` fully green

### Issues found during review

| # | Severity | File | Issue | Root cause |
|---|----------|------|-------|-----------|
| 1 | **High** | `graphs/bfs.zig:34` | `visited[neighbor]` panics on out-of-bounds neighbor index | No bounds check before indexing `visited[]` array |
| 2 | **High** | `graphs/dfs.zig:38` | `visited[neighbors[i]]` same panic | Same missing bounds check |
| 3 | **High** | `dynamic_programming/knapsack.zig:29` | `values[i-1]` panics when `weights.len != values.len` | No length-consistency validation (Python reference raises `ValueError`) |
| 4 | **Medium** | `graphs/bfs.zig:30` | `orderedRemove(0)` is O(n) head-delete, degrading BFS to O(V²+E) | Used ArrayList as naive queue instead of index-advancing pattern |

### Fix summary

| # | Fix | New test |
|---|-----|---------|
| 1 | `if (neighbor >= n) continue` — skip invalid neighbors | `bfs: invalid neighbor index is skipped` |
| 2 | `if (nb >= n) continue` — skip invalid neighbors | `dfs: invalid neighbor index is skipped` |
| 3 | `if (weights.len != values.len) return error.LengthMismatch` | `knapsack: length mismatch returns error` |
| 4 | Replace `orderedRemove(0)` with head-index advancing: `queue_buf.items[queue_head]; queue_head += 1` — O(1) dequeue | — (no new test, performance fix) |

---

## Cumulative Summary

| Metric | Phase 1 | + Batch 1 | + Batch 2A | + Batch 2B | + QA₁ | + Phase 3 | + QA₂ | **Total** |
|--------|---------|-----------|------------|------------|-------|-----------|-------|-----------|
| Algorithms | 5 | +15 | +6 | +7 | +0 | +8 | +0 | **41** |
| Test cases | 34 | +68 | +36 | +27 | +11 | +45 | +3 | **224** |
| First-attempt compile pass | 5/5 | 15/15 | 4/6 | 7/7 | N/A | 7/8 | N/A | **38/41 (92.7%)** |
| Manual fix lines | 0 | 0 | 2 | 0 | +424 | 1 | +8 | **435** |

### Key Observations

1. **Pure-logic algorithms translate cleanly.** No dynamic memory = no allocator hassle = high AI success rate.
2. **Zig idioms the AI got right:** `?usize` optionals, `comptime T: type` generics, `testing.expectEqualSlices`, `defer` for cleanup, `for (0..n)` range syntax, `@min`/`@intCast` builtins. In Phase 3: generic type-returning functions, `allocator.create/destroy` for linked nodes, `ArrayListUnmanaged` for heap backing.
3. **Post-implementation review remains essential.** Phase 3 surfaced 3 High-severity runtime panics on boundary inputs (BFS/DFS out-of-bounds, knapsack length mismatch) — none caught by the AI's own test cases.
4. **AI blind spot: defensive programming.** The AI consistently produces correct algorithms for valid inputs but rarely adds guards against malformed inputs. This pattern repeated across Phase 2 QA and Phase 3 QA.
5. **DFS compile failure was a Zig 0.15 API change** (`ArrayListUnmanaged.pop()` returns optional). This confirms the Phase 2 observation that AI training data lags behind Zig 0.15 API changes.
6. **All 41 algorithms implemented, all 224 tests green.** Project is feature-complete per the original plan.

---

# Vibe Coding 实验日志

> 用 AI 将 Python 算法翻译为 Zig (0.15.2)，记录成功率与报错模式。

## 第一阶段：Spike 验证（5 个算法）

**日期：** 2026-02-27
**模型：** Claude Opus 4.6
**Go/No-Go 结论：** Go（5/5 首次通过，阈值为 4/5）

---

### linear_search — 2026-02-27

- AI 编译尝试次数：**1**（首次即通过）
- 报错类型：无
- 人工修改行数：**0**
- 测试用例：7 个，全部通过
- 备注：最简单的算法。使用了 Zig 的 `for (items, 0..)` 索引迭代语法，简洁地道。

### bubble_sort — 2026-02-27

- AI 编译尝试次数：**1**（首次即通过）
- 报错类型：无
- 人工修改行数：**0**
- 测试用例：5 个，全部通过
- 备注：从 Python 直译。`while (i < n) : (i += 1)` 自然对应 Python 的 `for i in range(n)`。

### insertion_sort — 2026-02-27

- AI 编译尝试次数：**1**（首次即通过）
- 报错类型：无
- 人工修改行数：**0**
- 测试用例：6 个，全部通过
- 备注：Zig 关键差异：布尔逻辑用 `and` 而非 `&&`。内层循环的 `usize` 递减需注意，但 `j > 0` 守卫先求值，无需特殊处理。

### binary_search — 2026-02-27

- AI 编译尝试次数：**1**（首次即通过）
- 报错类型：无
- 人工修改行数：**0**
- 测试用例：9 个，全部通过
- 备注：返回 `?usize`（optional 类型）代替 Python 的 `-1` 惯例——更符合 Zig 风格。添加了 `mid == 0` 守卫防止 `high = mid - 1` 时 `usize` 下溢。

### merge_sort — 2026-02-27

- AI 编译尝试次数：**1**（首次即通过）
- 报错类型：无
- 人工修改行数：**0**
- 测试用例：7 个，全部通过
- 备注：第一个需要 `std.mem.Allocator` 的算法。测试中使用 `testing.allocator`（自动检测内存泄漏）。`defer allocator.free()` 模式实现类 RAII 资源清理。

---

## 第二阶段第一批：简单算法（15 个）

**日期：** 2026-02-27
**模型：** Claude Opus 4.6
**批次结果：** 15/15 首次通过 (100%)

### 排序（新增 5 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| selection_sort | 1 | 0 | 5 | 使用 Zig 0.15 的 `for (0..n)` 范围语法，比 `while` 更简洁 |
| shell_sort | 1 | 0 | 6 | Ciura 间隔序列作为编译期数组。`continue` 跳过大于数组长度的间隔 |
| counting_sort | 1 | 0 | 5 | 非泛型（仅 i32）——索引计算需要 `@intCast`。使用 allocator |
| cocktail_shaker_sort | 1 | 0 | 5 | 双向冒泡排序。从右到左遍历用 `while (j > start)` 避免 usize 下溢 |
| gnome_sort | 1 | 0 | 5 | 最简单的排序——约 15 行。无 range-for，纯 while 循环手动控制索引 |

### 查找（新增 2 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| jump_search | 1 | 0 | 6 | 使用 `math.sqrt` 计算块大小，`@min` 内建函数做边界检查 |
| ternary_search | 1 | 0 | 7 | 范围 < 3 时回退到线性扫描，避免 usize 除法边界问题 |

### 数学（新增 8 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| gcd | 1 | 0 | 3 | 通过 `@intCast` 处理负数输入，返回 u64 |
| lcm | 1 | 0 | 4 | `a / gcd(a,b) * b` 先除后乘避免溢出 |
| fibonacci | 1 | 0 | 3 | 迭代实现，返回 u64。简洁的 `for (2..n+1)` 范围 |
| prime_check | 1 | 0 | 4 | 6k±1 优化。`i * i <= n` 避免调用 sqrt |
| sieve_of_eratosthenes | 1 | 0 | 5 | 使用 allocator 分配 bool 筛和结果数组。`@memset` 初始化 |
| power | 1 | 0 | 3 | 提供 `power(i64)` 和 `powerMod(u64)` 两个版本 |
| factorial | 1 | 0 | 3 | 迭代实现。`u64` 可容纳到 20! (2.4×10¹⁸) |
| collatz_sequence | 1 | 0 | 4 | 两个 API：`collatzSteps`（仅计数）和 `collatzSequence`（填充缓冲区） |

---

## 第二阶段第二批 A：中等算法（#16-#21，共 6 个）

**日期：** 2026-02-27
**模型：** Codex GPT-5
**批次结果：** 已实现 6 个，4/6 首次编译通过

### 排序（新增 4 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| quick_sort | 1 | 0 | 6 | 原地 Lomuto 分区，递归切分 slice |
| heap_sort | 1 | 0 | 6 | 最大堆 + 下沉（sift-down），全程原地 |
| radix_sort | 1 | 0 | 6 | 负数/非负数拆分后，对无符号数组做基数排序 |
| bucket_sort | 1 | 1 | 6 | sqrt(n) 桶 + 桶内插入排序；用 `@divTrunc` 修复有符号整除 |

### 查找（新增 2 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| exponential_search | 1 | 0 | 6 | 指数扩边界后做区间二分 |
| interpolation_search | 1 | 1 | 6 | 探测位置公式；用 `@divTrunc` 修复有符号整除 |

---

## 第二阶段第二批 B：中等算法（#22-#28，共 7 个）

**日期：** 2026-02-27
**模型：** Codex GPT-5
**批次结果：** 7/7 首次编译通过 (100%)

### 数据结构（新增 2 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| stack | 1 | 0 | 3 | 泛型动态数组栈，push/pop 均摊 O(1) |
| queue | 1 | 0 | 3 | 泛型环形缓冲区队列，支持动态扩容 |

### 动态规划（新增 5 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| climbing_stairs | 1 | 0 | 3 | 经典一维 DP 递推，空间 O(1) |
| fibonacci_dp | 1 | 0 | 3 | 记忆化递归（`allocator` + memo 表） |
| coin_change | 1 | 0 | 5 | 一维 DP 求最少硬币，不可达返回 `null` |
| max_subarray_sum | 1 | 0 | 5 | Kadane 单次遍历求最大子数组和 |
| longest_common_subsequence | 1 | 0 | 5 | 二维 DP（扁平化分配）返回 LCS 长度 |

---

## 第二阶段 QA 修复批：安全性与语义对齐（无新增算法）

**日期：** 2026-02-27
**模型：** Codex GPT-5
**批次结果：** 已修复全部回归问题，`zig build test` 全绿

### 评审中发现的问题

- 边界输入下可触发运行时 panic：
  - `gcd/lcm` 在 `minInt(i64)` 上溢出
  - `powerMod` 在 `modulus == 0` 时除零
  - `counting_sort` 极大取值范围下溢出/内存爆炸
  - `collatz_sequence` 小缓冲区可能越界写入
- 与 Python 参考语义不一致：
  - `coin_change`：最少硬币数 vs 方案总数
  - `fibonacci_dp`：单值输出 vs 序列输出
  - `longest_common_subsequence`：仅长度 vs `(长度, 子序列)`
  - `max_subarray_sum`：空数组行为与 `allow_empty_subarrays` 模式
  - `climbing_stairs`：缺少正整数输入约束
  - `stack/queue`：缺少 Python 风格错误 API

### 修复摘要

- 为上述风险点补齐显式输入校验和错误返回。
- 将部分 DP 算法输出形式对齐至 Python 参考行为。
- 在保留现有易用接口的同时，增加 Python 风格错误 API（`StackOverflow/Underflow`、`EmptyQueue`）。
- 为每个问题新增定向回归测试。
- 人工修复量：12 个算法文件共 **+305 / -119** 行。

---

## 第三阶段：困难算法（#29-#36，共 8 个）

**日期：** 2026-02-27
**模型：** Claude Opus 4.6
**批次结果：** 7/8 首次编译通过（DFS 需 1 行修复）

### 数据结构（新增 4 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| singly_linked_list | 1 | 0 | 6 | 通过 `pub fn SinglyLinkedList(comptime T: type) type` 实现泛型。`allocator.create/destroy` 管理节点。`?*Node` 指针链 |
| doubly_linked_list | 1 | 0 | 7 | head+tail 双指针。`insertTail` O(1)。reverse 交换每个节点的 `prev`/`next` 并交换 head/tail |
| binary_search_tree | 1 | 0 | 6 | 递归插入/查找/删除。双子节点删除使用中序后继替换。`deinit` 递归释放子树 |
| min_heap | 1 | 0 | 5 | `ArrayListUnmanaged` 做底层数组。siftUp/siftDown。`fromSlice` 在 O(n) 内建堆 |

### 动态规划（新增 2 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| edit_distance | 1 | 0 | 6 | 扁平化二维 DP 表 `(m+1)×(n+1)`。自底向上，支持插入/删除/替换 |
| knapsack | 1 | 0 | 6 | 扁平化二维 DP 表 `(n+1)×(W+1)`。标准 0/1 取或不取递推 |

### 图算法（新增 2 个）

| 算法 | 尝试次数 | 人工修改 | 测试数 | 备注 |
|------|---------|---------|--------|------|
| bfs | 1 | 0 | 4 | ArrayList 做队列 + visited 数组。邻接表类型 `[]const []const usize` |
| dfs | **2** | **1** | 5 | `ArrayListUnmanaged.pop()` 在 Zig 0.15 中返回 `?usize`——AI 忘记 `.?` 解包。1 行修复 |

---

## 第三阶段 QA 修复批：边界安全加固

**日期：** 2026-02-27
**模型：** Claude Opus 4.6
**批次结果：** 修复 4 个问题，新增 3 个回归测试，`zig build test` 全绿

### 评审中发现的问题

| # | 严重度 | 文件 | 问题 | 根因 |
|---|--------|------|------|------|
| 1 | **高** | `graphs/bfs.zig:34` | `visited[neighbor]` 在邻接点越界时 panic | 索引 `visited[]` 前未校验 `neighbor < n` |
| 2 | **高** | `graphs/dfs.zig:38` | `visited[neighbors[i]]` 同样越界 panic | 同类缺失边界检查 |
| 3 | **高** | `dynamic_programming/knapsack.zig:29` | `weights.len != values.len` 时 `values[i-1]` 越界 panic | 未做长度一致性校验（Python 参考会抛 `ValueError`） |
| 4 | **中** | `graphs/bfs.zig:30` | `orderedRemove(0)` 为 O(n) 头删，BFS 整体退化至 O(V²+E) | 用 ArrayList 做朴素队列而非 index 前进模式 |

### 修复摘要

| # | 修复方式 | 新增测试 |
|---|---------|---------|
| 1 | `if (neighbor >= n) continue` — 跳过非法邻接点 | `bfs: invalid neighbor index is skipped` |
| 2 | `if (nb >= n) continue` — 跳过非法邻接点 | `dfs: invalid neighbor index is skipped` |
| 3 | `if (weights.len != values.len) return error.LengthMismatch` | `knapsack: length mismatch returns error` |
| 4 | `orderedRemove(0)` → head-index 前进：`queue_buf.items[queue_head]; queue_head += 1` — O(1) 出队 | — （性能修复，无新测试） |

---

## 累计统计

| 指标 | 第一阶段 | + 第一批 | + 第二批 A | + 第二批 B | + QA₁ | + 第三阶段 | + QA₂ | **合计** |
|------|---------|---------|-----------|-----------|-------|-----------|-------|---------|
| 算法数 | 5 | +15 | +6 | +7 | +0 | +8 | +0 | **41** |
| 测试用例 | 34 | +68 | +36 | +27 | +11 | +45 | +3 | **224** |
| 首次编译通过 | 5/5 | 15/15 | 4/6 | 7/7 | N/A | 7/8 | N/A | **38/41 (92.7%)** |
| 人工修改行数 | 0 | 0 | 2 | 0 | +424 | 1 | +8 | **435** |

### 关键观察

1. **纯逻辑算法翻译效果极好。** 无动态内存 = 无 allocator 麻烦 = AI 成功率高。
2. **AI 正确使用的 Zig 惯用法：** `?usize` optional 类型、`comptime T: type` 泛型、`testing.expectEqualSlices`、`defer` 资源清理、`for (0..n)` 范围语法、`@min`/`@intCast` 内建函数。第三阶段新增：泛型类型返回函数、`allocator.create/destroy` 管理链表节点、`ArrayListUnmanaged` 做堆底层。
3. **后置评审仍然不可或缺。** 第三阶段发现 3 个高严重度运行时 panic（BFS/DFS 越界、knapsack 长度不匹配）——均未被 AI 自身的测试用例覆盖。
4. **AI 盲区：防御性编程。** AI 始终能为合法输入生成正确算法，但几乎不主动为畸形输入添加守卫。这一模式在第二阶段 QA 和第三阶段 QA 中反复出现。
5. **DFS 编译失败源于 Zig 0.15 API 变更**（`ArrayListUnmanaged.pop()` 返回 optional）。再次验证了 AI 训练数据滞后于 Zig 0.15 API 的结论。
6. **全部 41 个算法已实现，224 个测试全绿。** 项目按原始计划完成全部功能。
