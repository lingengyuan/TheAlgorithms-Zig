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

## Cumulative Summary

| Metric | Phase 1 | + Batch 1 | **Total** |
|--------|---------|-----------|-----------|
| Algorithms | 5 | +15 | **20** |
| Test cases | 34 | +68 | **102** |
| First-attempt pass rate | 100% | 100% | **100%** |
| Manual fix lines | 0 | 0 | **0** |

### Key Observations

1. **Pure-logic algorithms translate cleanly.** No dynamic memory = no allocator hassle = high AI success rate.
2. **Zig idioms the AI got right:** `?usize` optionals, `comptime T: type` generics, `testing.expectEqualSlices`, `defer` for cleanup, `for (0..n)` range syntax, `@min`/`@intCast` builtins.
3. **Allocator-using algorithms also passed first try** (counting_sort, sieve_of_eratosthenes), suggesting Phase 2 Batch 2 (★★☆) is feasible.
4. **Anticipated challenge for Batch 2:** Quick sort partition logic, heap sort sift-down, and DP with 2D allocator tables.

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

## 累计统计

| 指标 | 第一阶段 | + 第一批 | **合计** |
|------|---------|---------|---------|
| 算法数 | 5 | +15 | **20** |
| 测试用例 | 34 | +68 | **102** |
| 首次通过率 | 100% | 100% | **100%** |
| 人工修改行数 | 0 | 0 | **0** |

### 关键观察

1. **纯逻辑算法翻译效果极好。** 无动态内存 = 无 allocator 麻烦 = AI 成功率高。
2. **AI 正确使用的 Zig 惯用法：** `?usize` optional 类型、`comptime T: type` 泛型、`testing.expectEqualSlices`、`defer` 资源清理、`for (0..n)` 范围语法、`@min`/`@intCast` 内建函数。
3. **需要 allocator 的算法也首次通过**（counting_sort、sieve_of_eratosthenes），表明第二批（★★☆）可行。
4. **第二批预期挑战：** 快速排序分区逻辑、堆排序下沉操作、2D allocator 表的动态规划。
