# TheAlgorithms - Zig

Classic algorithm implementations in Zig, with built-in unit tests. Inspired by [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python).

This project is also a **vibe coding experiment**: using AI to translate Python algorithms into Zig — a language the author has zero prior experience with — and recording success rates, failure patterns, and human intervention costs along the way.

---

## Features

- Each algorithm lives in a single `.zig` file with `test` blocks — no separate test files needed
- Comptime generics: most algorithms work across `i32`, `f64`, and other numeric types
- Zero dependencies beyond the Zig standard library
- Unified test runner: `zig build test` runs every algorithm's tests in one command
- Tested on **Zig 0.15.2**

## Algorithm Catalog

### Sorting (12)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Bubble Sort | [`sorts/bubble_sort.zig`](sorts/bubble_sort.zig) | O(n²) |
| Insertion Sort | [`sorts/insertion_sort.zig`](sorts/insertion_sort.zig) | O(n²) |
| Merge Sort | [`sorts/merge_sort.zig`](sorts/merge_sort.zig) | O(n log n) |
| Quick Sort | [`sorts/quick_sort.zig`](sorts/quick_sort.zig) | O(n log n) avg |
| Heap Sort | [`sorts/heap_sort.zig`](sorts/heap_sort.zig) | O(n log n) |
| Radix Sort | [`sorts/radix_sort.zig`](sorts/radix_sort.zig) | O(d · (n + b)) |
| Bucket Sort | [`sorts/bucket_sort.zig`](sorts/bucket_sort.zig) | O(n + k) avg |
| Selection Sort | [`sorts/selection_sort.zig`](sorts/selection_sort.zig) | O(n²) |
| Shell Sort | [`sorts/shell_sort.zig`](sorts/shell_sort.zig) | O(n^1.3) avg |
| Gnome Sort | [`sorts/gnome_sort.zig`](sorts/gnome_sort.zig) | O(n²) |
| Cocktail Shaker Sort | [`sorts/cocktail_shaker_sort.zig`](sorts/cocktail_shaker_sort.zig) | O(n²) |
| Counting Sort | [`sorts/counting_sort.zig`](sorts/counting_sort.zig) | O(n + k) |

### Searching (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Linear Search | [`searches/linear_search.zig`](searches/linear_search.zig) | O(n) |
| Binary Search | [`searches/binary_search.zig`](searches/binary_search.zig) | O(log n) |
| Exponential Search | [`searches/exponential_search.zig`](searches/exponential_search.zig) | O(log n) |
| Interpolation Search | [`searches/interpolation_search.zig`](searches/interpolation_search.zig) | O(log log n) avg |
| Jump Search | [`searches/jump_search.zig`](searches/jump_search.zig) | O(√n) |
| Ternary Search | [`searches/ternary_search.zig`](searches/ternary_search.zig) | O(log₃ n) |

### Math (8)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| GCD (Euclidean) | [`maths/gcd.zig`](maths/gcd.zig) | O(log n) |
| LCM | [`maths/lcm.zig`](maths/lcm.zig) | O(log n) |
| Fibonacci | [`maths/fibonacci.zig`](maths/fibonacci.zig) | O(n) |
| Factorial | [`maths/factorial.zig`](maths/factorial.zig) | O(n) |
| Prime Check | [`maths/prime_check.zig`](maths/prime_check.zig) | O(√n) |
| Sieve of Eratosthenes | [`maths/sieve_of_eratosthenes.zig`](maths/sieve_of_eratosthenes.zig) | O(n log log n) |
| Binary Exponentiation | [`maths/power.zig`](maths/power.zig) | O(log n) |
| Collatz Sequence | [`maths/collatz_sequence.zig`](maths/collatz_sequence.zig) | O(?) |

### Data Structures (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Stack (Array) | [`data_structures/stack.zig`](data_structures/stack.zig) | O(1) amortized push/pop |
| Queue (Array Circular Buffer) | [`data_structures/queue.zig`](data_structures/queue.zig) | O(1) amortized enqueue/dequeue |
| Singly Linked List | [`data_structures/singly_linked_list.zig`](data_structures/singly_linked_list.zig) | O(1) insert head, O(n) insert tail |
| Doubly Linked List | [`data_structures/doubly_linked_list.zig`](data_structures/doubly_linked_list.zig) | O(1) insert/delete head/tail |
| Binary Search Tree | [`data_structures/binary_search_tree.zig`](data_structures/binary_search_tree.zig) | O(log n) avg insert/search/delete |
| Min Heap | [`data_structures/min_heap.zig`](data_structures/min_heap.zig) | O(log n) insert/extract, O(n) heapify |

### Dynamic Programming (7)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Climbing Stairs | [`dynamic_programming/climbing_stairs.zig`](dynamic_programming/climbing_stairs.zig) | O(n) |
| Fibonacci (Memoized DP) | [`dynamic_programming/fibonacci_dp.zig`](dynamic_programming/fibonacci_dp.zig) | O(n) |
| Coin Change (Ways) | [`dynamic_programming/coin_change.zig`](dynamic_programming/coin_change.zig) | O(amount × coin_count) |
| Max Subarray Sum (Kadane) | [`dynamic_programming/max_subarray_sum.zig`](dynamic_programming/max_subarray_sum.zig) | O(n) |
| Longest Common Subsequence | [`dynamic_programming/longest_common_subsequence.zig`](dynamic_programming/longest_common_subsequence.zig) | O(m × n) |
| Edit Distance | [`dynamic_programming/edit_distance.zig`](dynamic_programming/edit_distance.zig) | O(m × n) |
| 0/1 Knapsack | [`dynamic_programming/knapsack.zig`](dynamic_programming/knapsack.zig) | O(n × W) |

### Graphs (2)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Breadth-First Search (BFS) | [`graphs/bfs.zig`](graphs/bfs.zig) | O(V + E) |
| Depth-First Search (DFS) | [`graphs/dfs.zig`](graphs/dfs.zig) | O(V + E) |

### Bit Manipulation (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Is Power of Two | [`bit_manipulation/is_power_of_two.zig`](bit_manipulation/is_power_of_two.zig) | O(1) |
| Count Set Bits | [`bit_manipulation/count_set_bits.zig`](bit_manipulation/count_set_bits.zig) | O(k) |
| Find Unique Number | [`bit_manipulation/find_unique_number.zig`](bit_manipulation/find_unique_number.zig) | O(n) |
| Reverse Bits | [`bit_manipulation/reverse_bits.zig`](bit_manipulation/reverse_bits.zig) | O(1) |
| Missing Number | [`bit_manipulation/missing_number.zig`](bit_manipulation/missing_number.zig) | O(n) |
| Is Power of Four | [`bit_manipulation/power_of_4.zig`](bit_manipulation/power_of_4.zig) | O(1) |

### Conversions (4)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Decimal to Binary | [`conversions/decimal_to_binary.zig`](conversions/decimal_to_binary.zig) | O(log n) |
| Binary to Decimal | [`conversions/binary_to_decimal.zig`](conversions/binary_to_decimal.zig) | O(n) |
| Decimal to Hexadecimal | [`conversions/decimal_to_hexadecimal.zig`](conversions/decimal_to_hexadecimal.zig) | O(log n) |
| Binary to Hexadecimal | [`conversions/binary_to_hexadecimal.zig`](conversions/binary_to_hexadecimal.zig) | O(n) |

### Strings (10)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Palindrome Check | [`strings/palindrome.zig`](strings/palindrome.zig) | O(n) |
| Reverse Words | [`strings/reverse_words.zig`](strings/reverse_words.zig) | O(n) |
| Anagram Check | [`strings/anagram.zig`](strings/anagram.zig) | O(n) |
| Hamming Distance | [`strings/hamming_distance.zig`](strings/hamming_distance.zig) | O(n) |
| Naive String Search | [`strings/naive_string_search.zig`](strings/naive_string_search.zig) | O(n·m) |
| Knuth-Morris-Pratt | [`strings/knuth_morris_pratt.zig`](strings/knuth_morris_pratt.zig) | O(n + m) |
| Rabin-Karp | [`strings/rabin_karp.zig`](strings/rabin_karp.zig) | O(n + m) avg |
| Z-Function | [`strings/z_function.zig`](strings/z_function.zig) | O(n + m) |
| Levenshtein Distance | [`strings/levenshtein_distance.zig`](strings/levenshtein_distance.zig) | O(m × n) |
| Is Pangram | [`strings/is_pangram.zig`](strings/is_pangram.zig) | O(n) |

## Quick Start

```bash
# Install Zig 0.15.2 (no root required)
# See https://ziglang.org/download/

# Run all tests
zig build test

# Run a single algorithm's tests
zig test sorts/bubble_sort.zig
```

## Project Structure

```
TheAlgorithms-Zig/
├── build.zig                # Build script — registers all test files
├── build.zig.zon            # Package manifest
├── sorts/                   # 12 sorting algorithms
├── searches/                # 6 search algorithms
├── maths/                   # 8 math algorithms
├── data_structures/         # 6 data structure implementations
├── dynamic_programming/     # 7 dynamic programming algorithms
├── graphs/                  # 2 graph traversal algorithms
├── bit_manipulation/        # 6 bit manipulation algorithms
├── conversions/             # 4 number base conversions
└── strings/                 # 10 string algorithms
```

## Development

**Requirements:** Zig ≥ 0.15.2

Each algorithm file is self-contained: implementation + tests in one file. To add a new algorithm:

1. Create `<category>/<algorithm_name>.zig`
2. Implement the algorithm as a `pub fn` with comptime generics where appropriate
3. Add `test` blocks at the bottom of the file
4. Register the file in `build.zig`'s `test_files` array
5. Run `zig build test` to verify

## Vibe Coding Experiment

This project doubles as a research experiment on AI-assisted development. Every algorithm records:

- How many AI attempts were needed to produce compilable code
- What error categories appeared (type inference, allocator, comptime, etc.)
- How many lines of manual human fixes were required

Results will be published in `EXPERIMENT_LOG.md` as the project progresses.

## Contributing

Contributions welcome! Please ensure:

- [ ] `zig build test` passes
- [ ] Each file includes a reference comment linking to the Python source
- [ ] Complexity is documented in the doc comment

## License

MIT

---

# TheAlgorithms - Zig（简体中文）

经典算法的 Zig 实现，每个算法内置单元测试。灵感来自 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)。

本项目同时是一个 **vibe coding 实验**：用 AI 将 Python 算法翻译为 Zig——一门作者此前零基础的语言——并记录 AI 的成功率、报错模式和人工干预成本。

---

## 功能特性

- 每个算法独立为一个 `.zig` 文件，实现与测试写在同一文件中
- 使用 comptime 泛型，大多数算法支持 `i32`、`f64` 等数值类型
- 零外部依赖，仅使用 Zig 标准库
- 统一测试入口：`zig build test` 一键运行所有算法测试
- 基于 **Zig 0.15.2** 测试通过

## 算法目录

### 排序 (12)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 冒泡排序 | [`sorts/bubble_sort.zig`](sorts/bubble_sort.zig) | O(n²) |
| 插入排序 | [`sorts/insertion_sort.zig`](sorts/insertion_sort.zig) | O(n²) |
| 归并排序 | [`sorts/merge_sort.zig`](sorts/merge_sort.zig) | O(n log n) |
| 快速排序 | [`sorts/quick_sort.zig`](sorts/quick_sort.zig) | O(n log n) 平均 |
| 堆排序 | [`sorts/heap_sort.zig`](sorts/heap_sort.zig) | O(n log n) |
| 基数排序 | [`sorts/radix_sort.zig`](sorts/radix_sort.zig) | O(d · (n + b)) |
| 桶排序 | [`sorts/bucket_sort.zig`](sorts/bucket_sort.zig) | O(n + k) 平均 |
| 选择排序 | [`sorts/selection_sort.zig`](sorts/selection_sort.zig) | O(n²) |
| 希尔排序 | [`sorts/shell_sort.zig`](sorts/shell_sort.zig) | O(n^1.3) 平均 |
| 侏儒排序 | [`sorts/gnome_sort.zig`](sorts/gnome_sort.zig) | O(n²) |
| 鸡尾酒排序 | [`sorts/cocktail_shaker_sort.zig`](sorts/cocktail_shaker_sort.zig) | O(n²) |
| 计数排序 | [`sorts/counting_sort.zig`](sorts/counting_sort.zig) | O(n + k) |

### 查找 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 线性查找 | [`searches/linear_search.zig`](searches/linear_search.zig) | O(n) |
| 二分查找 | [`searches/binary_search.zig`](searches/binary_search.zig) | O(log n) |
| 指数查找 | [`searches/exponential_search.zig`](searches/exponential_search.zig) | O(log n) |
| 插值查找 | [`searches/interpolation_search.zig`](searches/interpolation_search.zig) | O(log log n) 平均 |
| 跳跃查找 | [`searches/jump_search.zig`](searches/jump_search.zig) | O(√n) |
| 三分查找 | [`searches/ternary_search.zig`](searches/ternary_search.zig) | O(log₃ n) |

### 数学 (8)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 最大公约数 | [`maths/gcd.zig`](maths/gcd.zig) | O(log n) |
| 最小公倍数 | [`maths/lcm.zig`](maths/lcm.zig) | O(log n) |
| 斐波那契数列 | [`maths/fibonacci.zig`](maths/fibonacci.zig) | O(n) |
| 阶乘 | [`maths/factorial.zig`](maths/factorial.zig) | O(n) |
| 素数判定 | [`maths/prime_check.zig`](maths/prime_check.zig) | O(√n) |
| 埃拉托色尼筛法 | [`maths/sieve_of_eratosthenes.zig`](maths/sieve_of_eratosthenes.zig) | O(n log log n) |
| 快速幂 | [`maths/power.zig`](maths/power.zig) | O(log n) |
| 考拉兹序列 | [`maths/collatz_sequence.zig`](maths/collatz_sequence.zig) | O(?) |

### 数据结构 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 栈（数组实现） | [`data_structures/stack.zig`](data_structures/stack.zig) | push/pop 均摊 O(1) |
| 队列（数组环形缓冲区） | [`data_structures/queue.zig`](data_structures/queue.zig) | enqueue/dequeue 均摊 O(1) |
| 单向链表 | [`data_structures/singly_linked_list.zig`](data_structures/singly_linked_list.zig) | 头插 O(1)，尾插 O(n) |
| 双向链表 | [`data_structures/doubly_linked_list.zig`](data_structures/doubly_linked_list.zig) | 头尾插删 O(1) |
| 二叉搜索树 | [`data_structures/binary_search_tree.zig`](data_structures/binary_search_tree.zig) | 插入/查找/删除平均 O(log n) |
| 最小堆 | [`data_structures/min_heap.zig`](data_structures/min_heap.zig) | 插入/取出 O(log n)，建堆 O(n) |

### 动态规划 (7)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 爬楼梯 | [`dynamic_programming/climbing_stairs.zig`](dynamic_programming/climbing_stairs.zig) | O(n) |
| 斐波那契（记忆化 DP） | [`dynamic_programming/fibonacci_dp.zig`](dynamic_programming/fibonacci_dp.zig) | O(n) |
| 硬币找零（方案数） | [`dynamic_programming/coin_change.zig`](dynamic_programming/coin_change.zig) | O(amount × coin_count) |
| 最大子数组和（Kadane） | [`dynamic_programming/max_subarray_sum.zig`](dynamic_programming/max_subarray_sum.zig) | O(n) |
| 最长公共子序列 | [`dynamic_programming/longest_common_subsequence.zig`](dynamic_programming/longest_common_subsequence.zig) | O(m × n) |
| 编辑距离 | [`dynamic_programming/edit_distance.zig`](dynamic_programming/edit_distance.zig) | O(m × n) |
| 0/1 背包 | [`dynamic_programming/knapsack.zig`](dynamic_programming/knapsack.zig) | O(n × W) |

### 图算法 (2)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 广度优先搜索 (BFS) | [`graphs/bfs.zig`](graphs/bfs.zig) | O(V + E) |
| 深度优先搜索 (DFS) | [`graphs/dfs.zig`](graphs/dfs.zig) | O(V + E) |

### 位运算 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 判断 2 的幂 | [`bit_manipulation/is_power_of_two.zig`](bit_manipulation/is_power_of_two.zig) | O(1) |
| 统计置位数 | [`bit_manipulation/count_set_bits.zig`](bit_manipulation/count_set_bits.zig) | O(k) |
| 找唯一数 | [`bit_manipulation/find_unique_number.zig`](bit_manipulation/find_unique_number.zig) | O(n) |
| 位翻转 | [`bit_manipulation/reverse_bits.zig`](bit_manipulation/reverse_bits.zig) | O(1) |
| 缺失数字 | [`bit_manipulation/missing_number.zig`](bit_manipulation/missing_number.zig) | O(n) |
| 判断 4 的幂 | [`bit_manipulation/power_of_4.zig`](bit_manipulation/power_of_4.zig) | O(1) |

### 进制转换 (4)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 十进制转二进制 | [`conversions/decimal_to_binary.zig`](conversions/decimal_to_binary.zig) | O(log n) |
| 二进制转十进制 | [`conversions/binary_to_decimal.zig`](conversions/binary_to_decimal.zig) | O(n) |
| 十进制转十六进制 | [`conversions/decimal_to_hexadecimal.zig`](conversions/decimal_to_hexadecimal.zig) | O(log n) |
| 二进制转十六进制 | [`conversions/binary_to_hexadecimal.zig`](conversions/binary_to_hexadecimal.zig) | O(n) |

### 字符串 (10)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 回文检查 | [`strings/palindrome.zig`](strings/palindrome.zig) | O(n) |
| 单词反转 | [`strings/reverse_words.zig`](strings/reverse_words.zig) | O(n) |
| 异位词检查 | [`strings/anagram.zig`](strings/anagram.zig) | O(n) |
| 汉明距离 | [`strings/hamming_distance.zig`](strings/hamming_distance.zig) | O(n) |
| 朴素字符串搜索 | [`strings/naive_string_search.zig`](strings/naive_string_search.zig) | O(n·m) |
| KMP 字符串搜索 | [`strings/knuth_morris_pratt.zig`](strings/knuth_morris_pratt.zig) | O(n + m) |
| Rabin-Karp | [`strings/rabin_karp.zig`](strings/rabin_karp.zig) | O(n + m) 平均 |
| Z 函数 | [`strings/z_function.zig`](strings/z_function.zig) | O(n + m) |
| Levenshtein 距离 | [`strings/levenshtein_distance.zig`](strings/levenshtein_distance.zig) | O(m × n) |
| 全字母句检查 | [`strings/is_pangram.zig`](strings/is_pangram.zig) | O(n) |

## 快速开始

```bash
# 安装 Zig 0.15.2（无需 root 权限）
# 参见 https://ziglang.org/download/

# 运行所有测试
zig build test

# 运行单个算法的测试
zig test sorts/bubble_sort.zig
```

## 项目结构

```
TheAlgorithms-Zig/
├── build.zig                # 构建脚本 — 注册所有测试文件
├── build.zig.zon            # 包清单
├── sorts/                   # 12 种排序算法
├── searches/                # 6 种查找算法
├── maths/                   # 8 种数学算法
├── data_structures/         # 6 种数据结构实现
├── dynamic_programming/     # 7 个动态规划算法
├── graphs/                  # 2 个图遍历算法
├── bit_manipulation/        # 6 个位运算算法
├── conversions/             # 4 个进制转换
└── strings/                 # 10 个字符串算法
```

## 开发指南

**环境要求：** Zig ≥ 0.15.2

每个算法文件自包含：实现 + 测试写在同一文件。添加新算法的步骤：

1. 创建 `<分类>/<算法名>.zig`
2. 用 `pub fn` 实现算法，适当使用 comptime 泛型
3. 在文件末尾添加 `test` 块
4. 在 `build.zig` 的 `test_files` 数组中注册该文件
5. 运行 `zig build test` 验证

## Vibe Coding 实验

本项目同时是一项 AI 辅助开发的研究实验。每个算法会记录：

- AI 需要几次尝试才能生成可编译的代码
- 出现了哪些报错类别（类型推断、内存分配、comptime 语法等）
- 需要人工手动修改多少行

实验结果将在 `EXPERIMENT_LOG.md` 中持续更新。

## 贡献指南

欢迎贡献！请确保：

- [ ] `zig build test` 通过
- [ ] 每个文件包含指向 Python 源代码的参考注释
- [ ] 文档注释中标注了时间/空间复杂度

## 许可证

MIT
