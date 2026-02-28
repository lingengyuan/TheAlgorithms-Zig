# Contributing to TheAlgorithms-Zig

Thank you for your interest in contributing! This project ports classic algorithms from [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) to Zig.

## Prerequisites

- **Zig 0.15.2** — download from [ziglang.org](https://ziglang.org/download/)
- Familiarity with the algorithm you want to implement
- Reference implementation from TheAlgorithms/Python (or Wikipedia for algorithms without a Python equivalent)

## How to Add an Algorithm

### 1. Create the file

Place your `.zig` file in the appropriate category directory:

```
sorts/           searches/         maths/
data_structures/ dynamic_programming/ graphs/
bit_manipulation/ conversions/      greedy_methods/
matrix/          backtracking/      strings/
```

Every file must start with a reference header:

```zig
//! Algorithm Name - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/<path>.py
```

### 2. Write self-contained code

Each file contains both the implementation and tests. Follow these conventions:

- Use `comptime T: type` generics where the algorithm is type-agnostic
- Use `testing.allocator` for heap allocations (it detects leaks)
- `defer allocator.free(...)` for all allocations within a function scope
- No `&&` or `||` — Zig uses `and` / `or`
- Guard against unsigned integer subtraction underflow

### 3. Register in `build.zig`

Add your file to the `test_files` array under the correct category comment.

### 4. Run tests

```bash
zig build test
```

All existing tests plus your new tests must pass.

### 5. Update documentation

- **README.md** — add a row to the algorithm table in both the English and Chinese sections; update the category count
- **EXPERIMENT_LOG.md** — record your batch entry in both languages with compile attempts, test results, and notes

### 6. Submit a PR

- One algorithm per file
- Descriptive commit message following the project convention
- All tests green

## Code Style

- Keep files self-contained — no cross-file imports between algorithm files
- Prefer clarity over cleverness
- Include edge case tests (empty input, single element, duplicates, etc.)

## Zig 0.15 Gotchas

| Trap | Correct Usage |
|------|--------------|
| `build.zig` test setup | Use `root_module = b.createModule(...)` inside `addTest` |
| `ArrayListUnmanaged.pop()` | Returns `?T` — unwrap with `.?` or `orelse` |
| `ArrayListUnmanaged` methods | All take `allocator` as first argument |
| Boolean operators | `and` / `or` (not `&&` / `||`) |
| Unsigned subtraction | Guard before subtracting to avoid debug-mode underflow |

---

# 贡献指南

感谢你对 TheAlgorithms-Zig 的关注！本项目将 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) 中的经典算法移植到 Zig。

## 前置条件

- **Zig 0.15.2** — 从 [ziglang.org](https://ziglang.org/download/) 下载
- 熟悉你要实现的算法
- TheAlgorithms/Python 中的参考实现（没有 Python 对应版本的算法可使用 Wikipedia）

## 如何添加算法

### 1. 创建文件

将 `.zig` 文件放在对应的分类目录中：

```
sorts/           searches/         maths/
data_structures/ dynamic_programming/ graphs/
bit_manipulation/ conversions/      greedy_methods/
matrix/          backtracking/      strings/
```

每个文件必须以引用头开始：

```zig
//! 算法名称 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/<path>.py
```

### 2. 编写自包含代码

每个文件同时包含实现和测试，遵循以下规范：

- 对类型无关的算法使用 `comptime T: type` 泛型
- 测试中使用 `testing.allocator` 进行堆分配（可检测内存泄漏）
- 函数作用域内的所有分配使用 `defer allocator.free(...)`
- 不使用 `&&` 或 `||` — Zig 使用 `and` / `or`
- 对无符号整数减法进行溢出保护

### 3. 在 `build.zig` 中注册

将文件路径添加到 `test_files` 数组中对应分类的注释下方。

### 4. 运行测试

```bash
zig build test
```

所有现有测试和新测试都必须通过。

### 5. 更新文档

- **README.md** — 在英文和中文两个部分的算法表中各添加一行，更新分类计数
- **EXPERIMENT_LOG.md** — 在两种语言中记录批次条目，包括编译尝试次数、测试结果和备注

### 6. 提交 PR

- 每个文件一个算法
- 描述性的提交信息，遵循项目提交规范
- 所有测试通过

## 代码风格

- 文件自包含 — 算法文件之间不互相导入
- 清晰优先于巧妙
- 包含边界情况测试（空输入、单元素、重复元素等）

## Zig 0.15 注意事项

| 陷阱 | 正确用法 |
|------|---------|
| `build.zig` 测试配置 | 在 `addTest` 中使用 `root_module = b.createModule(...)` |
| `ArrayListUnmanaged.pop()` | 返回 `?T` — 用 `.?` 或 `orelse` 解包 |
| `ArrayListUnmanaged` 方法 | 所有方法都需要 `allocator` 作为第一个参数 |
| 布尔运算符 | `and` / `or`（不是 `&&` / `||`） |
| 无符号减法 | 减法前先检查以避免调试模式下的溢出 |
