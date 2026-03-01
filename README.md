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

### Math (16)

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
| Extended Euclidean | [`maths/extended_euclidean.zig`](maths/extended_euclidean.zig) | O(log n) |
| Modular Inverse | [`maths/modular_inverse.zig`](maths/modular_inverse.zig) | O(log m) |
| Euler's Totient | [`maths/eulers_totient.zig`](maths/eulers_totient.zig) | O(√n) |
| Chinese Remainder Theorem | [`maths/chinese_remainder_theorem.zig`](maths/chinese_remainder_theorem.zig) | O(k² + k log M) |
| Binomial Coefficient | [`maths/binomial_coefficient.zig`](maths/binomial_coefficient.zig) | O(min(k, n-k)) |
| Integer Square Root | [`maths/integer_square_root.zig`](maths/integer_square_root.zig) | O(log n) |
| Miller-Rabin Primality Test | [`maths/miller_rabin.zig`](maths/miller_rabin.zig) | O(k · log³ n), k=7 |
| Matrix Exponentiation | [`maths/matrix_exponentiation.zig`](maths/matrix_exponentiation.zig) | O(n³ log p) |

### Data Structures (17)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Stack (Array) | [`data_structures/stack.zig`](data_structures/stack.zig) | O(1) amortized push/pop |
| Queue (Array Circular Buffer) | [`data_structures/queue.zig`](data_structures/queue.zig) | O(1) amortized enqueue/dequeue |
| Singly Linked List | [`data_structures/singly_linked_list.zig`](data_structures/singly_linked_list.zig) | O(1) insert head, O(n) insert tail |
| Doubly Linked List | [`data_structures/doubly_linked_list.zig`](data_structures/doubly_linked_list.zig) | O(1) insert/delete head/tail |
| Binary Search Tree | [`data_structures/binary_search_tree.zig`](data_structures/binary_search_tree.zig) | O(log n) avg insert/search/delete |
| Min Heap | [`data_structures/min_heap.zig`](data_structures/min_heap.zig) | O(log n) insert/extract, O(n) heapify |
| Trie | [`data_structures/trie.zig`](data_structures/trie.zig) | O(L) insert/search/delete |
| Disjoint Set (Union-Find) | [`data_structures/disjoint_set.zig`](data_structures/disjoint_set.zig) | O(alpha(n)) amortized |
| AVL Tree | [`data_structures/avl_tree.zig`](data_structures/avl_tree.zig) | O(log n) insert/search/delete |
| Max Heap | [`data_structures/max_heap.zig`](data_structures/max_heap.zig) | O(log n) insert/extract, O(n) heapify |
| Priority Queue | [`data_structures/priority_queue.zig`](data_structures/priority_queue.zig) | O(log n) enqueue/dequeue |
| Hash Map (Open Addressing) | [`data_structures/hash_map_open_addressing.zig`](data_structures/hash_map_open_addressing.zig) | O(1) avg put/get/remove |
| Segment Tree (Range Max Query) | [`data_structures/segment_tree.zig`](data_structures/segment_tree.zig) | O(log n) query/update |
| Fenwick Tree (Binary Indexed Tree) | [`data_structures/fenwick_tree.zig`](data_structures/fenwick_tree.zig) | O(log n) add/prefix/range |
| Red-Black Tree | [`data_structures/red_black_tree.zig`](data_structures/red_black_tree.zig) | O(log n) insert/search |
| LRU Cache | [`data_structures/lru_cache.zig`](data_structures/lru_cache.zig) | O(1) avg get/put |
| Deque (Ring Buffer) | [`data_structures/deque.zig`](data_structures/deque.zig) | O(1) amortized push/pop both ends |

### Dynamic Programming (17)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Climbing Stairs | [`dynamic_programming/climbing_stairs.zig`](dynamic_programming/climbing_stairs.zig) | O(n) |
| Fibonacci (Memoized DP) | [`dynamic_programming/fibonacci_dp.zig`](dynamic_programming/fibonacci_dp.zig) | O(n) |
| Coin Change (Ways) | [`dynamic_programming/coin_change.zig`](dynamic_programming/coin_change.zig) | O(amount × coin_count) |
| Max Subarray Sum (Kadane) | [`dynamic_programming/max_subarray_sum.zig`](dynamic_programming/max_subarray_sum.zig) | O(n) |
| Longest Increasing Subsequence | [`dynamic_programming/longest_increasing_subsequence.zig`](dynamic_programming/longest_increasing_subsequence.zig) | O(n log n) |
| Rod Cutting | [`dynamic_programming/rod_cutting.zig`](dynamic_programming/rod_cutting.zig) | O(n²) |
| Matrix Chain Multiplication | [`dynamic_programming/matrix_chain_multiplication.zig`](dynamic_programming/matrix_chain_multiplication.zig) | O(n³) |
| Palindrome Partitioning (Min Cuts) | [`dynamic_programming/palindrome_partitioning.zig`](dynamic_programming/palindrome_partitioning.zig) | O(n²) |
| Word Break | [`dynamic_programming/word_break.zig`](dynamic_programming/word_break.zig) | O(n·m·k) |
| Catalan Numbers | [`dynamic_programming/catalan_numbers.zig`](dynamic_programming/catalan_numbers.zig) | O(n²) |
| Longest Common Subsequence | [`dynamic_programming/longest_common_subsequence.zig`](dynamic_programming/longest_common_subsequence.zig) | O(m × n) |
| Edit Distance | [`dynamic_programming/edit_distance.zig`](dynamic_programming/edit_distance.zig) | O(m × n) |
| 0/1 Knapsack | [`dynamic_programming/knapsack.zig`](dynamic_programming/knapsack.zig) | O(n × W) |
| Subset Sum | [`dynamic_programming/subset_sum.zig`](dynamic_programming/subset_sum.zig) | O(n × target) |
| Egg Drop Problem | [`dynamic_programming/egg_drop_problem.zig`](dynamic_programming/egg_drop_problem.zig) | O(eggs × answer) |
| Longest Palindromic Subsequence | [`dynamic_programming/longest_palindromic_subsequence.zig`](dynamic_programming/longest_palindromic_subsequence.zig) | O(n²) |
| Maximum Product Subarray | [`dynamic_programming/max_product_subarray.zig`](dynamic_programming/max_product_subarray.zig) | O(n) |

### Graphs (16)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Breadth-First Search (BFS) | [`graphs/bfs.zig`](graphs/bfs.zig) | O(V + E) |
| Depth-First Search (DFS) | [`graphs/dfs.zig`](graphs/dfs.zig) | O(V + E) |
| Dijkstra Shortest Path | [`graphs/dijkstra.zig`](graphs/dijkstra.zig) | O(V² + E) |
| A* Search | [`graphs/a_star_search.zig`](graphs/a_star_search.zig) | O(V² + E) |
| Tarjan SCC | [`graphs/tarjan_scc.zig`](graphs/tarjan_scc.zig) | O(V + E) |
| Bridges (Articulation Edges) | [`graphs/bridges.zig`](graphs/bridges.zig) | O(V + E) |
| Eulerian Path/Circuit (Undirected) | [`graphs/eulerian_path_circuit_undirected.zig`](graphs/eulerian_path_circuit_undirected.zig) | O(V + E) |
| Ford-Fulkerson Max Flow | [`graphs/ford_fulkerson.zig`](graphs/ford_fulkerson.zig) | O(V · E²) |
| Bipartite Check (BFS) | [`graphs/bipartite_check_bfs.zig`](graphs/bipartite_check_bfs.zig) | O(V + E) |
| Bellman-Ford Shortest Path | [`graphs/bellman_ford.zig`](graphs/bellman_ford.zig) | O(V·E) |
| Topological Sort | [`graphs/topological_sort.zig`](graphs/topological_sort.zig) | O(V + E) |
| Floyd-Warshall | [`graphs/floyd_warshall.zig`](graphs/floyd_warshall.zig) | O(V³) |
| Detect Cycle (Directed) | [`graphs/detect_cycle.zig`](graphs/detect_cycle.zig) | O(V + E) |
| Connected Components | [`graphs/connected_components.zig`](graphs/connected_components.zig) | O(V + E) |
| Kruskal MST | [`graphs/kruskal.zig`](graphs/kruskal.zig) | O(E log E) |
| Prim MST | [`graphs/prim.zig`](graphs/prim.zig) | O(V² + E) |

### Greedy Methods (7)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Best Time to Buy/Sell Stock | [`greedy_methods/best_time_to_buy_sell_stock.zig`](greedy_methods/best_time_to_buy_sell_stock.zig) | O(n) |
| Minimum Coin Change (Greedy) | [`greedy_methods/minimum_coin_change.zig`](greedy_methods/minimum_coin_change.zig) | O(n·k) |
| Minimum Waiting Time | [`greedy_methods/minimum_waiting_time.zig`](greedy_methods/minimum_waiting_time.zig) | O(n log n) |
| Fractional Knapsack | [`greedy_methods/fractional_knapsack.zig`](greedy_methods/fractional_knapsack.zig) | O(n log n) |
| Activity Selection | [`greedy_methods/activity_selection.zig`](greedy_methods/activity_selection.zig) | O(n) |
| Huffman Coding | [`greedy_methods/huffman_coding.zig`](greedy_methods/huffman_coding.zig) | O(n + σ log σ) |
| Job Sequencing with Deadlines | [`greedy_methods/job_sequencing_with_deadline.zig`](greedy_methods/job_sequencing_with_deadline.zig) | O(n log n + n·d) |

### Matrix (5)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Matrix Multiply | [`matrix/matrix_multiply.zig`](matrix/matrix_multiply.zig) | O(m·k·n) |
| Matrix Transpose | [`matrix/matrix_transpose.zig`](matrix/matrix_transpose.zig) | O(m·n) |
| Rotate Matrix 90° | [`matrix/rotate_matrix.zig`](matrix/rotate_matrix.zig) | O(n²) |
| Spiral Print | [`matrix/spiral_print.zig`](matrix/spiral_print.zig) | O(m·n) |
| Pascal's Triangle | [`matrix/pascal_triangle.zig`](matrix/pascal_triangle.zig) | O(n²) |

### Backtracking (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Permutations | [`backtracking/permutations.zig`](backtracking/permutations.zig) | O(n! · n) |
| Combinations | [`backtracking/combinations.zig`](backtracking/combinations.zig) | O(C(n,k)) |
| Subsets | [`backtracking/subsets.zig`](backtracking/subsets.zig) | O(2ⁿ) |
| Generate Parentheses | [`backtracking/generate_parentheses.zig`](backtracking/generate_parentheses.zig) | O(Catalan(n)) |
| N-Queens | [`backtracking/n_queens.zig`](backtracking/n_queens.zig) | O(n!) |
| Sudoku Solver | [`backtracking/sudoku_solver.zig`](backtracking/sudoku_solver.zig) | O(9^m) |

### Bit Manipulation (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Is Power of Two | [`bit_manipulation/is_power_of_two.zig`](bit_manipulation/is_power_of_two.zig) | O(1) |
| Count Set Bits | [`bit_manipulation/count_set_bits.zig`](bit_manipulation/count_set_bits.zig) | O(k) |
| Find Unique Number | [`bit_manipulation/find_unique_number.zig`](bit_manipulation/find_unique_number.zig) | O(n) |
| Reverse Bits | [`bit_manipulation/reverse_bits.zig`](bit_manipulation/reverse_bits.zig) | O(1) |
| Missing Number | [`bit_manipulation/missing_number.zig`](bit_manipulation/missing_number.zig) | O(n) |
| Is Power of Four | [`bit_manipulation/power_of_4.zig`](bit_manipulation/power_of_4.zig) | O(1) |

### Conversions (7)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Decimal to Binary | [`conversions/decimal_to_binary.zig`](conversions/decimal_to_binary.zig) | O(log n) |
| Binary to Decimal | [`conversions/binary_to_decimal.zig`](conversions/binary_to_decimal.zig) | O(n) |
| Decimal to Hexadecimal | [`conversions/decimal_to_hexadecimal.zig`](conversions/decimal_to_hexadecimal.zig) | O(log n) |
| Binary to Hexadecimal | [`conversions/binary_to_hexadecimal.zig`](conversions/binary_to_hexadecimal.zig) | O(n) |
| Roman to Integer | [`conversions/roman_to_integer.zig`](conversions/roman_to_integer.zig) | O(n) |
| Integer to Roman | [`conversions/integer_to_roman.zig`](conversions/integer_to_roman.zig) | O(1) bounded range |
| Temperature Conversion | [`conversions/temperature_conversion.zig`](conversions/temperature_conversion.zig) | O(1) |

### Ciphers (1)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Caesar Cipher | [`ciphers/caesar_cipher.zig`](ciphers/caesar_cipher.zig) | O(n · m) |

### Hashing (1)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| SHA-256 | [`hashing/sha256.zig`](hashing/sha256.zig) | O(n) |

### Strings (13)

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
| Aho-Corasick | [`strings/aho_corasick.zig`](strings/aho_corasick.zig) | O(text + matches) query |
| Suffix Array | [`strings/suffix_array.zig`](strings/suffix_array.zig) | O(n log² n) build |
| Run-Length Encoding | [`strings/run_length_encoding.zig`](strings/run_length_encoding.zig) | O(n) encode/decode |

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
├── maths/                   # 16 math algorithms
├── data_structures/         # 17 data structure implementations
├── dynamic_programming/     # 17 dynamic programming algorithms
├── graphs/                  # 16 graph algorithms
├── bit_manipulation/        # 6 bit manipulation algorithms
├── conversions/             # 7 number base conversions
├── ciphers/                 # 1 cipher algorithm
├── hashing/                 # 1 hashing algorithm
├── strings/                 # 13 string algorithms
├── greedy_methods/          # 7 greedy algorithms
├── matrix/                  # 5 matrix algorithms
└── backtracking/            # 6 backtracking algorithms
```

## Development

**Requirements:** Zig ≥ 0.15.2

Each algorithm file is self-contained: implementation + tests in one file. To add a new algorithm:

1. Create `<category>/<algorithm_name>.zig`
2. Implement the algorithm as a `pub fn` with comptime generics where appropriate
3. Add `test` blocks at the bottom of the file
4. Register the file in `build.zig`'s `test_files` array
5. Run `zig build test` to verify

## Python vs Zig Benchmark (Alignable Set)

As of **March 1, 2026**, this repository includes a one-click benchmark that compares Python and Zig implementations on a shared workload.

- Benchmark scope: **130 alignable algorithms** out of 130 total
- Current data-structure subset included in benchmark harness: `stack`, `queue`, `singly_linked_list`, `doubly_linked_list`, `binary_search_tree`, `min_heap`, `max_heap`, `priority_queue`, `trie`, `disjoint_set`, `avl_tree`, `hash_map_open_addressing`, `segment_tree`, `fenwick_tree`, `red_black_tree`, `lru_cache`, `deque`
- Current backtracking subset included in benchmark harness: `permutations`, `combinations`, `subsets`, `generate_parentheses`, `n_queens`, `sudoku_solver`
- Environment used for the latest numbers:
  - Zig `0.15.2`
  - Python `3.12.3`
  - CPU: `Intel(R) Xeon(R) Platinum 8255C`, 2 vCPU

Summary from the latest run:

| Metric | Value |
|---|---:|
| Alignable algorithms benchmarked | 130 |
| Checksum match count | 130 |
| Mean speedup (Python/Zig) | 161.74x |
| Median speedup (Python/Zig) | 26.91x |
| Geometric mean speedup (Python/Zig) | 20.49x |

Run everything with one command:

```bash
bash benchmarks/python_vs_zig/run_all.sh
```

Run a single algorithm and merge it into full benchmark artifacts:

```bash
bash benchmarks/python_vs_zig/run_single.sh dijkstra
```

Generated outputs:

- Full leaderboard (Markdown): [`benchmarks/python_vs_zig/leaderboard_all.md`](benchmarks/python_vs_zig/leaderboard_all.md)
- Full leaderboard (CSV): [`benchmarks/python_vs_zig/leaderboard_all.csv`](benchmarks/python_vs_zig/leaderboard_all.csv)
- Category chart data (CSV): [`benchmarks/python_vs_zig/category_speedup_chart.csv`](benchmarks/python_vs_zig/category_speedup_chart.csv)
- Run summary (Markdown): [`benchmarks/python_vs_zig/summary_all.md`](benchmarks/python_vs_zig/summary_all.md)

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

### 数学 (16)

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
| 扩展欧几里得算法 | [`maths/extended_euclidean.zig`](maths/extended_euclidean.zig) | O(log n) |
| 模逆 | [`maths/modular_inverse.zig`](maths/modular_inverse.zig) | O(log m) |
| 欧拉函数（Totient） | [`maths/eulers_totient.zig`](maths/eulers_totient.zig) | O(√n) |
| 中国剩余定理 | [`maths/chinese_remainder_theorem.zig`](maths/chinese_remainder_theorem.zig) | O(k² + k log M) |
| 二项式系数 | [`maths/binomial_coefficient.zig`](maths/binomial_coefficient.zig) | O(min(k, n-k)) |
| 整数平方根 | [`maths/integer_square_root.zig`](maths/integer_square_root.zig) | O(log n) |
| Miller-Rabin 素性测试 | [`maths/miller_rabin.zig`](maths/miller_rabin.zig) | O(k · log³ n)，k=7 |
| 矩阵快速幂 | [`maths/matrix_exponentiation.zig`](maths/matrix_exponentiation.zig) | O(n³ log p) |

### 数据结构 (17)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 栈（数组实现） | [`data_structures/stack.zig`](data_structures/stack.zig) | push/pop 均摊 O(1) |
| 队列（数组环形缓冲区） | [`data_structures/queue.zig`](data_structures/queue.zig) | enqueue/dequeue 均摊 O(1) |
| 单向链表 | [`data_structures/singly_linked_list.zig`](data_structures/singly_linked_list.zig) | 头插 O(1)，尾插 O(n) |
| 双向链表 | [`data_structures/doubly_linked_list.zig`](data_structures/doubly_linked_list.zig) | 头尾插删 O(1) |
| 二叉搜索树 | [`data_structures/binary_search_tree.zig`](data_structures/binary_search_tree.zig) | 插入/查找/删除平均 O(log n) |
| 最小堆 | [`data_structures/min_heap.zig`](data_structures/min_heap.zig) | 插入/取出 O(log n)，建堆 O(n) |
| Trie（前缀树） | [`data_structures/trie.zig`](data_structures/trie.zig) | 插入/查询/删除 O(L) |
| 并查集（Union-Find） | [`data_structures/disjoint_set.zig`](data_structures/disjoint_set.zig) | 均摊 O(alpha(n)) |
| AVL 树 | [`data_structures/avl_tree.zig`](data_structures/avl_tree.zig) | 插入/查找/删除 O(log n) |
| 最大堆 | [`data_structures/max_heap.zig`](data_structures/max_heap.zig) | 插入/取出 O(log n)，建堆 O(n) |
| 优先队列 | [`data_structures/priority_queue.zig`](data_structures/priority_queue.zig) | 入队/出队 O(log n) |
| 开放寻址哈希表 | [`data_structures/hash_map_open_addressing.zig`](data_structures/hash_map_open_addressing.zig) | put/get/remove 平均 O(1) |
| 线段树（区间最大值） | [`data_structures/segment_tree.zig`](data_structures/segment_tree.zig) | 查询/更新 O(log n) |
| 树状数组（Fenwick Tree） | [`data_structures/fenwick_tree.zig`](data_structures/fenwick_tree.zig) | add/prefix/range O(log n) |
| 红黑树 | [`data_structures/red_black_tree.zig`](data_structures/red_black_tree.zig) | 插入/查找 O(log n) |
| LRU 缓存 | [`data_structures/lru_cache.zig`](data_structures/lru_cache.zig) | get/put 平均 O(1) |
| 双端队列（环形缓冲） | [`data_structures/deque.zig`](data_structures/deque.zig) | 两端 push/pop 均摊 O(1) |

### 动态规划 (17)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 爬楼梯 | [`dynamic_programming/climbing_stairs.zig`](dynamic_programming/climbing_stairs.zig) | O(n) |
| 斐波那契（记忆化 DP） | [`dynamic_programming/fibonacci_dp.zig`](dynamic_programming/fibonacci_dp.zig) | O(n) |
| 硬币找零（方案数） | [`dynamic_programming/coin_change.zig`](dynamic_programming/coin_change.zig) | O(amount × coin_count) |
| 最大子数组和（Kadane） | [`dynamic_programming/max_subarray_sum.zig`](dynamic_programming/max_subarray_sum.zig) | O(n) |
| 最长递增子序列 | [`dynamic_programming/longest_increasing_subsequence.zig`](dynamic_programming/longest_increasing_subsequence.zig) | O(n log n) |
| 钢条切割 | [`dynamic_programming/rod_cutting.zig`](dynamic_programming/rod_cutting.zig) | O(n²) |
| 矩阵链乘法 | [`dynamic_programming/matrix_chain_multiplication.zig`](dynamic_programming/matrix_chain_multiplication.zig) | O(n³) |
| 回文划分（最少切割） | [`dynamic_programming/palindrome_partitioning.zig`](dynamic_programming/palindrome_partitioning.zig) | O(n²) |
| 单词拆分 | [`dynamic_programming/word_break.zig`](dynamic_programming/word_break.zig) | O(n·m·k) |
| Catalan 数 | [`dynamic_programming/catalan_numbers.zig`](dynamic_programming/catalan_numbers.zig) | O(n²) |
| 最长公共子序列 | [`dynamic_programming/longest_common_subsequence.zig`](dynamic_programming/longest_common_subsequence.zig) | O(m × n) |
| 编辑距离 | [`dynamic_programming/edit_distance.zig`](dynamic_programming/edit_distance.zig) | O(m × n) |
| 0/1 背包 | [`dynamic_programming/knapsack.zig`](dynamic_programming/knapsack.zig) | O(n × W) |
| 子集和 | [`dynamic_programming/subset_sum.zig`](dynamic_programming/subset_sum.zig) | O(n × target) |
| 鸡蛋掉落问题 | [`dynamic_programming/egg_drop_problem.zig`](dynamic_programming/egg_drop_problem.zig) | O(eggs × answer) |
| 最长回文子序列 | [`dynamic_programming/longest_palindromic_subsequence.zig`](dynamic_programming/longest_palindromic_subsequence.zig) | O(n²) |
| 最大乘积子数组 | [`dynamic_programming/max_product_subarray.zig`](dynamic_programming/max_product_subarray.zig) | O(n) |

### 图算法 (16)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 广度优先搜索 (BFS) | [`graphs/bfs.zig`](graphs/bfs.zig) | O(V + E) |
| 深度优先搜索 (DFS) | [`graphs/dfs.zig`](graphs/dfs.zig) | O(V + E) |
| Dijkstra 最短路径 | [`graphs/dijkstra.zig`](graphs/dijkstra.zig) | O(V² + E) |
| A* 搜索 | [`graphs/a_star_search.zig`](graphs/a_star_search.zig) | O(V² + E) |
| Tarjan 强连通分量 | [`graphs/tarjan_scc.zig`](graphs/tarjan_scc.zig) | O(V + E) |
| 桥边（割边） | [`graphs/bridges.zig`](graphs/bridges.zig) | O(V + E) |
| 欧拉路径/回路（无向图） | [`graphs/eulerian_path_circuit_undirected.zig`](graphs/eulerian_path_circuit_undirected.zig) | O(V + E) |
| Ford-Fulkerson 最大流 | [`graphs/ford_fulkerson.zig`](graphs/ford_fulkerson.zig) | O(V · E²) |
| 二分图检查（BFS） | [`graphs/bipartite_check_bfs.zig`](graphs/bipartite_check_bfs.zig) | O(V + E) |
| Bellman-Ford 最短路径 | [`graphs/bellman_ford.zig`](graphs/bellman_ford.zig) | O(V·E) |
| 拓扑排序 | [`graphs/topological_sort.zig`](graphs/topological_sort.zig) | O(V + E) |
| Floyd-Warshall | [`graphs/floyd_warshall.zig`](graphs/floyd_warshall.zig) | O(V³) |
| 有向图环检测 | [`graphs/detect_cycle.zig`](graphs/detect_cycle.zig) | O(V + E) |
| 连通分量计数 | [`graphs/connected_components.zig`](graphs/connected_components.zig) | O(V + E) |
| Kruskal 最小生成树 | [`graphs/kruskal.zig`](graphs/kruskal.zig) | O(E log E) |
| Prim 最小生成树 | [`graphs/prim.zig`](graphs/prim.zig) | O(V² + E) |

### 贪心算法 (7)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 买卖股票最佳时机 | [`greedy_methods/best_time_to_buy_sell_stock.zig`](greedy_methods/best_time_to_buy_sell_stock.zig) | O(n) |
| 最少硬币数（贪心） | [`greedy_methods/minimum_coin_change.zig`](greedy_methods/minimum_coin_change.zig) | O(n·k) |
| 最小等待时间 | [`greedy_methods/minimum_waiting_time.zig`](greedy_methods/minimum_waiting_time.zig) | O(n log n) |
| 分数背包 | [`greedy_methods/fractional_knapsack.zig`](greedy_methods/fractional_knapsack.zig) | O(n log n) |
| 活动选择 | [`greedy_methods/activity_selection.zig`](greedy_methods/activity_selection.zig) | O(n) |
| 哈夫曼编码 | [`greedy_methods/huffman_coding.zig`](greedy_methods/huffman_coding.zig) | O(n + σ log σ) |
| 截止时间作业调度 | [`greedy_methods/job_sequencing_with_deadline.zig`](greedy_methods/job_sequencing_with_deadline.zig) | O(n log n + n·d) |

### 矩阵 (5)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 矩阵乘法 | [`matrix/matrix_multiply.zig`](matrix/matrix_multiply.zig) | O(m·k·n) |
| 矩阵转置 | [`matrix/matrix_transpose.zig`](matrix/matrix_transpose.zig) | O(m·n) |
| 矩阵旋转 90° | [`matrix/rotate_matrix.zig`](matrix/rotate_matrix.zig) | O(n²) |
| 螺旋打印 | [`matrix/spiral_print.zig`](matrix/spiral_print.zig) | O(m·n) |
| 杨辉三角 | [`matrix/pascal_triangle.zig`](matrix/pascal_triangle.zig) | O(n²) |

### 回溯算法 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 全排列 | [`backtracking/permutations.zig`](backtracking/permutations.zig) | O(n! · n) |
| 组合 | [`backtracking/combinations.zig`](backtracking/combinations.zig) | O(C(n,k)) |
| 子集（幂集） | [`backtracking/subsets.zig`](backtracking/subsets.zig) | O(2ⁿ) |
| 生成括号 | [`backtracking/generate_parentheses.zig`](backtracking/generate_parentheses.zig) | O(Catalan(n)) |
| N 皇后 | [`backtracking/n_queens.zig`](backtracking/n_queens.zig) | O(n!) |
| 数独求解 | [`backtracking/sudoku_solver.zig`](backtracking/sudoku_solver.zig) | O(9^m) |

### 位运算 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 判断 2 的幂 | [`bit_manipulation/is_power_of_two.zig`](bit_manipulation/is_power_of_two.zig) | O(1) |
| 统计置位数 | [`bit_manipulation/count_set_bits.zig`](bit_manipulation/count_set_bits.zig) | O(k) |
| 找唯一数 | [`bit_manipulation/find_unique_number.zig`](bit_manipulation/find_unique_number.zig) | O(n) |
| 位翻转 | [`bit_manipulation/reverse_bits.zig`](bit_manipulation/reverse_bits.zig) | O(1) |
| 缺失数字 | [`bit_manipulation/missing_number.zig`](bit_manipulation/missing_number.zig) | O(n) |
| 判断 4 的幂 | [`bit_manipulation/power_of_4.zig`](bit_manipulation/power_of_4.zig) | O(1) |

### 进制转换 (7)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 十进制转二进制 | [`conversions/decimal_to_binary.zig`](conversions/decimal_to_binary.zig) | O(log n) |
| 二进制转十进制 | [`conversions/binary_to_decimal.zig`](conversions/binary_to_decimal.zig) | O(n) |
| 十进制转十六进制 | [`conversions/decimal_to_hexadecimal.zig`](conversions/decimal_to_hexadecimal.zig) | O(log n) |
| 二进制转十六进制 | [`conversions/binary_to_hexadecimal.zig`](conversions/binary_to_hexadecimal.zig) | O(n) |
| 罗马数字转整数 | [`conversions/roman_to_integer.zig`](conversions/roman_to_integer.zig) | O(n) |
| 整数转罗马数字 | [`conversions/integer_to_roman.zig`](conversions/integer_to_roman.zig) | O(1)（有界区间） |
| 温度单位转换 | [`conversions/temperature_conversion.zig`](conversions/temperature_conversion.zig) | O(1) |

### 密码学 (1)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 凯撒密码 | [`ciphers/caesar_cipher.zig`](ciphers/caesar_cipher.zig) | O(n · m) |

### 哈希 (1)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| SHA-256 | [`hashing/sha256.zig`](hashing/sha256.zig) | O(n) |

### 字符串 (13)

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
| Aho-Corasick 多模式匹配 | [`strings/aho_corasick.zig`](strings/aho_corasick.zig) | 查询 O(text + matches) |
| 后缀数组 | [`strings/suffix_array.zig`](strings/suffix_array.zig) | 构建 O(n log² n) |
| 游程编码（RLE） | [`strings/run_length_encoding.zig`](strings/run_length_encoding.zig) | 编码/解码 O(n) |

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
├── maths/                   # 16 种数学算法
├── data_structures/         # 17 种数据结构实现
├── dynamic_programming/     # 17 个动态规划算法
├── graphs/                  # 16 个图算法
├── bit_manipulation/        # 6 个位运算算法
├── conversions/             # 7 个进制转换
├── ciphers/                 # 1 个密码学算法
├── hashing/                 # 1 个哈希算法
├── strings/                 # 13 个字符串算法
├── greedy_methods/          # 7 个贪心算法
├── matrix/                  # 5 个矩阵算法
└── backtracking/            # 6 个回溯算法
```

## 开发指南

**环境要求：** Zig ≥ 0.15.2

每个算法文件自包含：实现 + 测试写在同一文件。添加新算法的步骤：

1. 创建 `<分类>/<算法名>.zig`
2. 用 `pub fn` 实现算法，适当使用 comptime 泛型
3. 在文件末尾添加 `test` 块
4. 在 `build.zig` 的 `test_files` 数组中注册该文件
5. 运行 `zig build test` 验证

## Python vs Zig 性能对比（可对齐集合）

截至 **2026 年 3 月 1 日**，仓库已提供一键脚本，对 Python 与 Zig 在同一工作负载下做性能对比。

- 对比范围：130 个算法中的 **130 个可对齐算法**
- 当前已纳入基准的数据结构子集：`stack`、`queue`、`singly_linked_list`、`doubly_linked_list`、`binary_search_tree`、`min_heap`、`max_heap`、`priority_queue`、`trie`、`disjoint_set`、`avl_tree`、`hash_map_open_addressing`、`segment_tree`、`fenwick_tree`、`red_black_tree`、`lru_cache`、`deque`
- 当前已纳入基准的回溯算法子集：`permutations`、`combinations`、`subsets`、`generate_parentheses`、`n_queens`、`sudoku_solver`
- 本轮数据环境：
  - Zig `0.15.2`
  - Python `3.12.3`
  - CPU：`Intel(R) Xeon(R) Platinum 8255C`，2 vCPU

最新一轮汇总：

| 指标 | 数值 |
|---|---:|
| 已对齐并完成基准的算法数 | 130 |
| checksum 一致算法数 | 130 |
| 平均加速比（Python/Zig） | 161.74x |
| 中位数加速比（Python/Zig） | 26.91x |
| 几何平均加速比（Python/Zig） | 20.49x |

一键运行：

```bash
bash benchmarks/python_vs_zig/run_all.sh
```

单算法增量运行并合并进总数据：

```bash
bash benchmarks/python_vs_zig/run_single.sh dijkstra
```

输出文件：

- 完整总榜（Markdown）：[`benchmarks/python_vs_zig/leaderboard_all.md`](benchmarks/python_vs_zig/leaderboard_all.md)
- 完整总榜（CSV）：[`benchmarks/python_vs_zig/leaderboard_all.csv`](benchmarks/python_vs_zig/leaderboard_all.csv)
- 分类图表数据（CSV）：[`benchmarks/python_vs_zig/category_speedup_chart.csv`](benchmarks/python_vs_zig/category_speedup_chart.csv)
- 运行摘要（Markdown）：[`benchmarks/python_vs_zig/summary_all.md`](benchmarks/python_vs_zig/summary_all.md)

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
