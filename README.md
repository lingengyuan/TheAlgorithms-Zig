# TheAlgorithms - Zig

Classic algorithm implementations in Zig, with built-in unit tests. Inspired by [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python).

This project is also a **vibe coding experiment**: using AI to translate Python algorithms into Zig and recording correctness, failure modes, and maintenance costs along the way.

Phase 6 accounting note (2026-03-10): the portable target has been fully closed out. [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig) currently registers `925` algorithms; under the corrected per-category caps, `916` count toward the portable target, leaving `0` planned algorithms remaining.

---

## English

### Project Description

This repository implements algorithms and data-structure exercises in Zig, with each algorithm stored in a single `.zig` file that includes its own tests. The current behavior reference is [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python).

### Features

- Single-file algorithm modules with colocated `test` blocks
- Zero external runtime dependencies beyond the Zig standard library
- `zig build test` registry through [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig)
- Bilingual repository-level documentation and experiment logs
- Phase 6 portable target completed: `916 / 916`

### Quick Start

```bash
zig version
zig test sorts/bubble_sort.zig
zig build test
```

Tested toolchain: `Zig 0.15.2`

### Repository Status

- Registered algorithms: `925`
- Effective completed algorithms: `916`
- Remaining planned gap: `0`
- `Project Euler` implementations: `133`

### Algorithm Catalog

The full catalog has been split out of this root README.

- [Catalog Index](docs/algorithm_catalog/README.md)
- [Core Algorithms](docs/algorithm_catalog/core_algorithms.md)
- [Structures And Optimization](docs/algorithm_catalog/structures_and_optimization.md)
- [Applied Science](docs/algorithm_catalog/applied_science.md)
- [Text Security And Data](docs/algorithm_catalog/text_security_and_data.md)
- [Special Topics](docs/algorithm_catalog/special_topics.md)

### Development And Testing

- Add every new algorithm file to [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig)
- Run file-level tests during implementation: `zig test <file>`
- Run full-suite verification before commit/push: `zig build test`
- Keep experiment-log entries bilingual and truthful

### Project Structure

```text
├── sorts/                   # 50 sorting algorithms
├── searches/                # 16 searching algorithms
├── maths/                   # 144 math algorithms
├── data_structures/         # 101 data structure algorithms
├── dynamic_programming/     # 54 dynamic programming algorithms
├── graphs/                  # 82 graph algorithms
├── bit_manipulation/        # 27 bit manipulation algorithms
├── conversions/             # 27 conversion algorithms
├── strings/                 # 59 string algorithms
├── greedy_methods/          # 8 greedy algorithms
├── matrix/                  # 20 matrix algorithms
├── backtracking/            # 21 backtracking algorithms
├── project_euler/           # 133 Project Euler algorithms
├── ciphers/                 # 47 cipher algorithms
├── data_compression/        # 8 data compression algorithms
├── geodesy/                 # 2 geodesy algorithms
├── geometry/                # 1 geometry algorithm
├── knapsack/                # 3 knapsack algorithms
└── audio_filters/           # 2 audio-filter algorithms
```

### Maintenance Notes

- Root `README.md` stays concise and acts as the project overview plus index.
- Detailed algorithm listings live under `docs/algorithm_catalog/`.
- Experiment logs are indexed from [`EXPERIMENT_LOG.md`](/root/projects/TheAlgorithms-Zig/EXPERIMENT_LOG.md).

## 简体中文

### 项目说明

本仓库使用 Zig 实现算法与数据结构练习；每个算法放在单独的 `.zig` 文件中，并且在同文件内包含测试。当前行为对齐参考项目为 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)。

### 功能特性

- 单文件算法模块，测试与实现同置
- 除 Zig 标准库外无额外运行时依赖
- 通过 [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig) 统一注册 `zig build test`
- 仓库级文档与实验日志保持双语
- Phase 6 可移植目标已完成：`916 / 916`

### 快速开始

```bash
zig version
zig test sorts/bubble_sort.zig
zig build test
```

当前验证工具链：`Zig 0.15.2`

### 仓库状态

- 已注册算法数：`925`
- 有效完成算法数：`916`
- 剩余计划缺口：`0`
- `Project Euler` 实现数：`133`

### 算法目录

完整算法目录已经从根 README 中拆分出去。

- [目录索引](docs/algorithm_catalog/README.md)
- [核心算法](docs/algorithm_catalog/core_algorithms.md)
- [数据结构与优化方法](docs/algorithm_catalog/structures_and_optimization.md)
- [应用科学](docs/algorithm_catalog/applied_science.md)
- [文本、安全与数据](docs/algorithm_catalog/text_security_and_data.md)
- [专题算法](docs/algorithm_catalog/special_topics.md)

### 开发与测试

- 每新增一个算法文件，都要同步注册到 [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig)
- 实现过程中执行文件级测试：`zig test <file>`
- commit / push 前执行全量验证：`zig build test`
- 实验日志必须保持双语且如实记录错误

### 项目结构

```text
├── sorts/                   # 50 个排序算法
├── searches/                # 16 个查找算法
├── maths/                   # 144 个数学算法
├── data_structures/         # 101 个数据结构算法
├── dynamic_programming/     # 54 个动态规划算法
├── graphs/                  # 82 个图算法
├── bit_manipulation/        # 27 个位运算算法
├── conversions/             # 27 个进制转换算法
├── strings/                 # 59 个字符串算法
├── greedy_methods/          # 8 个贪心算法
├── matrix/                  # 20 个矩阵算法
├── backtracking/            # 21 个回溯算法
├── project_euler/           # 133 个 Project Euler 算法
├── ciphers/                 # 47 个密码学算法
├── data_compression/        # 8 个数据压缩算法
├── geodesy/                 # 2 个测地学算法
├── geometry/                # 1 个几何算法
├── knapsack/                # 3 个背包算法
└── audio_filters/           # 2 个音频滤波算法
```

### 维护说明

- 根 `README.md` 保持精简，只承担项目概览和索引角色。
- 详细算法目录统一放在 `docs/algorithm_catalog/` 下。
- 实验日志从 [`EXPERIMENT_LOG.md`](/root/projects/TheAlgorithms-Zig/EXPERIMENT_LOG.md) 进入索引。
