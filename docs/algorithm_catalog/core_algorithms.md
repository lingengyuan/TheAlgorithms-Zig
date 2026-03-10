# Core Algorithms / 核心算法

- Source of truth: the detailed catalog sections from the pre-split root README.
- 数据来源：拆分前根 README 的详细目录条目。

### Sorting (50)

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
| Binary Insertion Sort | [`sorts/binary_insertion_sort.zig`](sorts/binary_insertion_sort.zig) | O(n²) |
| Bogo Sort | [`sorts/bogo_sort.zig`](sorts/bogo_sort.zig) | Expected O((n+1)!) |
| Comb Sort | [`sorts/comb_sort.zig`](sorts/comb_sort.zig) | O(n²) worst |
| Cycle Sort | [`sorts/cycle_sort.zig`](sorts/cycle_sort.zig) | O(n²) |
| Double Sort | [`sorts/double_sort.zig`](sorts/double_sort.zig) | O(n²) |
| Odd-Even Sort | [`sorts/odd_even_sort.zig`](sorts/odd_even_sort.zig) | O(n²) |
| Pancake Sort | [`sorts/pancake_sort.zig`](sorts/pancake_sort.zig) | O(n²) |
| Recursive Insertion Sort | [`sorts/recursive_insertion_sort.zig`](sorts/recursive_insertion_sort.zig) | O(n²) |
| Stooge Sort | [`sorts/stooge_sort.zig`](sorts/stooge_sort.zig) | O(n^2.7095) |
| Wiggle Sort | [`sorts/wiggle_sort.zig`](sorts/wiggle_sort.zig) | O(n) |
| Bead Sort | [`sorts/bead_sort.zig`](sorts/bead_sort.zig) | O(n²) |
| Cyclic Sort | [`sorts/cyclic_sort.zig`](sorts/cyclic_sort.zig) | O(n) |
| Exchange Sort | [`sorts/exchange_sort.zig`](sorts/exchange_sort.zig) | O(n²) |
| Iterative Merge Sort | [`sorts/iterative_merge_sort.zig`](sorts/iterative_merge_sort.zig) | O(n log n) |
| Pigeon Sort | [`sorts/pigeon_sort.zig`](sorts/pigeon_sort.zig) | O(n + range) |
| Pigeonhole Sort | [`sorts/pigeonhole_sort.zig`](sorts/pigeonhole_sort.zig) | O(n + range) |
| Quick Sort (3-way Partition) | [`sorts/quick_sort_3_partition.zig`](sorts/quick_sort_3_partition.zig) | O(n log n) avg |
| Recursive Quick Sort | [`sorts/recursive_quick_sort.zig`](sorts/recursive_quick_sort.zig) | O(n log n) avg |
| Shrink Shell Sort | [`sorts/shrink_shell_sort.zig`](sorts/shrink_shell_sort.zig) | Sub-quadratic avg |
| Stalin Sort | [`sorts/stalin_sort.zig`](sorts/stalin_sort.zig) | O(n) |
| Bitonic Sort | [`sorts/bitonic_sort.zig`](sorts/bitonic_sort.zig) | O(n log² n) |
| Circle Sort | [`sorts/circle_sort.zig`](sorts/circle_sort.zig) | O(n log n) avg |
| Dutch National Flag Sort | [`sorts/dutch_national_flag_sort.zig`](sorts/dutch_national_flag_sort.zig) | O(n) |
| Odd-Even Transposition Sort | [`sorts/odd_even_transposition_single_threaded.zig`](sorts/odd_even_transposition_single_threaded.zig) | O(n²) |
| Recursive Merge Sort Array | [`sorts/recursive_mergesort_array.zig`](sorts/recursive_mergesort_array.zig) | O(n log n) |
| SlowSort | [`sorts/slowsort.zig`](sorts/slowsort.zig) | Super-polynomial |
| Strand Sort | [`sorts/strand_sort.zig`](sorts/strand_sort.zig) | O(n²) avg |
| Tree Sort | [`sorts/tree_sort.zig`](sorts/tree_sort.zig) | O(n log n) avg |
| Unknown Sort | [`sorts/unknown_sort.zig`](sorts/unknown_sort.zig) | O(n²) |
| Merge-Insertion Sort | [`sorts/merge_insertion_sort.zig`](sorts/merge_insertion_sort.zig) | O(n log n) |
| External Sort (In-Memory Blocks) | [`sorts/external_sort.zig`](sorts/external_sort.zig) | O(n log b + n·b) |
| IntroSort | [`sorts/intro_sort.zig`](sorts/intro_sort.zig) | O(n log n) |
| MSD Radix Sort | [`sorts/msd_radix_sort.zig`](sorts/msd_radix_sort.zig) | O(w·n) |
| Natural Sort | [`sorts/natural_sort.zig`](sorts/natural_sort.zig) | O(n log n · m) |
| Odd-Even Transposition Sort (Parallel Model) | [`sorts/odd_even_transposition_parallel.zig`](sorts/odd_even_transposition_parallel.zig) | O(n²) |
| Patience Sort | [`sorts/patience_sort.zig`](sorts/patience_sort.zig) | O(n log n) |
| Tim Sort (Educational Variant) | [`sorts/tim_sort.zig`](sorts/tim_sort.zig) | O(n log n) |
| Topological Sort | [`sorts/topological_sort.zig`](sorts/topological_sort.zig) | O(V + E) |

### 排序 (50)

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
| 二分插入排序 | [`sorts/binary_insertion_sort.zig`](sorts/binary_insertion_sort.zig) | O(n²) |
| Bogo 排序 | [`sorts/bogo_sort.zig`](sorts/bogo_sort.zig) | 期望 O((n+1)!) |
| 梳排序 | [`sorts/comb_sort.zig`](sorts/comb_sort.zig) | 最坏 O(n²) |
| 循环排序 | [`sorts/cycle_sort.zig`](sorts/cycle_sort.zig) | O(n²) |
| 双向冒泡排序 | [`sorts/double_sort.zig`](sorts/double_sort.zig) | O(n²) |
| 奇偶排序 | [`sorts/odd_even_sort.zig`](sorts/odd_even_sort.zig) | O(n²) |
| 煎饼排序 | [`sorts/pancake_sort.zig`](sorts/pancake_sort.zig) | O(n²) |
| 递归插入排序 | [`sorts/recursive_insertion_sort.zig`](sorts/recursive_insertion_sort.zig) | O(n²) |
| Stooge 排序 | [`sorts/stooge_sort.zig`](sorts/stooge_sort.zig) | O(n^2.7095) |
| 摆动排序 | [`sorts/wiggle_sort.zig`](sorts/wiggle_sort.zig) | O(n) |
| 珠排序 | [`sorts/bead_sort.zig`](sorts/bead_sort.zig) | O(n²) |
| 循环置换排序 | [`sorts/cyclic_sort.zig`](sorts/cyclic_sort.zig) | O(n) |
| 交换排序 | [`sorts/exchange_sort.zig`](sorts/exchange_sort.zig) | O(n²) |
| 迭代归并排序 | [`sorts/iterative_merge_sort.zig`](sorts/iterative_merge_sort.zig) | O(n log n) |
| 鸽巢排序（Pigeon） | [`sorts/pigeon_sort.zig`](sorts/pigeon_sort.zig) | O(n + range) |
| 鸽巢排序（Pigeonhole） | [`sorts/pigeonhole_sort.zig`](sorts/pigeonhole_sort.zig) | O(n + range) |
| 三路快排 | [`sorts/quick_sort_3_partition.zig`](sorts/quick_sort_3_partition.zig) | 平均 O(n log n) |
| 递归快速排序 | [`sorts/recursive_quick_sort.zig`](sorts/recursive_quick_sort.zig) | 平均 O(n log n) |
| 收缩增量希尔排序 | [`sorts/shrink_shell_sort.zig`](sorts/shrink_shell_sort.zig) | 平均亚二次 |
| 斯大林排序 | [`sorts/stalin_sort.zig`](sorts/stalin_sort.zig) | O(n) |
| 双调排序 | [`sorts/bitonic_sort.zig`](sorts/bitonic_sort.zig) | O(n log² n) |
| 圈排序 | [`sorts/circle_sort.zig`](sorts/circle_sort.zig) | 平均 O(n log n) |
| 荷兰国旗排序 | [`sorts/dutch_national_flag_sort.zig`](sorts/dutch_national_flag_sort.zig) | O(n) |
| 奇偶换位排序 | [`sorts/odd_even_transposition_single_threaded.zig`](sorts/odd_even_transposition_single_threaded.zig) | O(n²) |
| 递归归并数组排序 | [`sorts/recursive_mergesort_array.zig`](sorts/recursive_mergesort_array.zig) | O(n log n) |
| SlowSort | [`sorts/slowsort.zig`](sorts/slowsort.zig) | 超多项式 |
| Strand 排序 | [`sorts/strand_sort.zig`](sorts/strand_sort.zig) | 平均 O(n²) |
| 树排序 | [`sorts/tree_sort.zig`](sorts/tree_sort.zig) | 平均 O(n log n) |
| Unknown 排序 | [`sorts/unknown_sort.zig`](sorts/unknown_sort.zig) | O(n²) |
| 归并-插入排序 | [`sorts/merge_insertion_sort.zig`](sorts/merge_insertion_sort.zig) | O(n log n) |
| 外部排序（内存分块版） | [`sorts/external_sort.zig`](sorts/external_sort.zig) | O(n log b + n·b) |
| IntroSort | [`sorts/intro_sort.zig`](sorts/intro_sort.zig) | O(n log n) |
| MSD 基数排序 | [`sorts/msd_radix_sort.zig`](sorts/msd_radix_sort.zig) | O(w·n) |
| 自然排序 | [`sorts/natural_sort.zig`](sorts/natural_sort.zig) | O(n log n · m) |
| 奇偶换位排序（并行模型） | [`sorts/odd_even_transposition_parallel.zig`](sorts/odd_even_transposition_parallel.zig) | O(n²) |
| Patience 排序 | [`sorts/patience_sort.zig`](sorts/patience_sort.zig) | O(n log n) |
| Tim 排序（教学变体） | [`sorts/tim_sort.zig`](sorts/tim_sort.zig) | O(n log n) |
| 拓扑排序 | [`sorts/topological_sort.zig`](sorts/topological_sort.zig) | O(V + E) |

### Searching (16)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Linear Search | [`searches/linear_search.zig`](searches/linear_search.zig) | O(n) |
| Binary Search | [`searches/binary_search.zig`](searches/binary_search.zig) | O(log n) |
| Exponential Search | [`searches/exponential_search.zig`](searches/exponential_search.zig) | O(log n) |
| Fibonacci Search | [`searches/fibonacci_search.zig`](searches/fibonacci_search.zig) | O(log n) |
| Hill Climbing | [`searches/hill_climbing.zig`](searches/hill_climbing.zig) | O(max_iter) |
| Interpolation Search | [`searches/interpolation_search.zig`](searches/interpolation_search.zig) | O(log log n) avg |
| Jump Search | [`searches/jump_search.zig`](searches/jump_search.zig) | O(√n) |
| Ternary Search | [`searches/ternary_search.zig`](searches/ternary_search.zig) | O(log₃ n) |
| Double Linear Search | [`searches/double_linear_search.zig`](searches/double_linear_search.zig) | O(n) |
| Double Linear Search (Recursive) | [`searches/double_linear_search_recursion.zig`](searches/double_linear_search_recursion.zig) | O(n) |
| Sentinel Linear Search | [`searches/sentinel_linear_search.zig`](searches/sentinel_linear_search.zig) | O(n) |
| Simple Binary Search | [`searches/simple_binary_search.zig`](searches/simple_binary_search.zig) | O(log n) |
| Quick Select | [`searches/quick_select.zig`](searches/quick_select.zig) | average O(n), worst O(n²) |
| Median of Medians | [`searches/median_of_medians.zig`](searches/median_of_medians.zig) | O(n) |
| Simulated Annealing | [`searches/simulated_annealing.zig`](searches/simulated_annealing.zig) | O(iterations · neighbors) |
| Tabu Search | [`searches/tabu_search.zig`](searches/tabu_search.zig) | O(iterations · n³) |

### 查找 (16)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 线性查找 | [`searches/linear_search.zig`](searches/linear_search.zig) | O(n) |
| 二分查找 | [`searches/binary_search.zig`](searches/binary_search.zig) | O(log n) |
| 指数查找 | [`searches/exponential_search.zig`](searches/exponential_search.zig) | O(log n) |
| Fibonacci 查找 | [`searches/fibonacci_search.zig`](searches/fibonacci_search.zig) | O(log n) |
| 爬山算法 | [`searches/hill_climbing.zig`](searches/hill_climbing.zig) | O(max_iter) |
| 插值查找 | [`searches/interpolation_search.zig`](searches/interpolation_search.zig) | O(log log n) 平均 |
| 跳跃查找 | [`searches/jump_search.zig`](searches/jump_search.zig) | O(√n) |
| 三分查找 | [`searches/ternary_search.zig`](searches/ternary_search.zig) | O(log₃ n) |
| 双向线性查找 | [`searches/double_linear_search.zig`](searches/double_linear_search.zig) | O(n) |
| 双向线性查找（递归） | [`searches/double_linear_search_recursion.zig`](searches/double_linear_search_recursion.zig) | O(n) |
| 哨兵线性查找 | [`searches/sentinel_linear_search.zig`](searches/sentinel_linear_search.zig) | O(n) |
| 简单双向二分查找 | [`searches/simple_binary_search.zig`](searches/simple_binary_search.zig) | O(log n) |
| Quick Select | [`searches/quick_select.zig`](searches/quick_select.zig) | 平均 O(n)，最坏 O(n²) |
| Median of Medians | [`searches/median_of_medians.zig`](searches/median_of_medians.zig) | O(n) |
| 模拟退火 | [`searches/simulated_annealing.zig`](searches/simulated_annealing.zig) | O(iterations · neighbors) |
| 禁忌搜索 | [`searches/tabu_search.zig`](searches/tabu_search.zig) | O(iterations · n³) |

### Math (144)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| GCD (Euclidean) | [`maths/gcd.zig`](maths/gcd.zig) | O(log n) |
| LCM | [`maths/lcm.zig`](maths/lcm.zig) | O(log n) |
| Least Common Multiple (Python Filename Variant) | [`maths/least_common_multiple.zig`](maths/least_common_multiple.zig) | Slow: O(lcm / max(a, b)); Fast: O(log n) |
| Fibonacci | [`maths/fibonacci.zig`](maths/fibonacci.zig) | O(n) |
| Factorial | [`maths/factorial.zig`](maths/factorial.zig) | O(n) |
| Prime Check | [`maths/prime_check.zig`](maths/prime_check.zig) | O(√n) |
| Sieve of Eratosthenes | [`maths/sieve_of_eratosthenes.zig`](maths/sieve_of_eratosthenes.zig) | O(n log log n) |
| Binary Exponentiation | [`maths/power.zig`](maths/power.zig) | O(log n) |
| Collatz Sequence | [`maths/collatz_sequence.zig`](maths/collatz_sequence.zig) | O(?) |
| Extended Euclidean | [`maths/extended_euclidean.zig`](maths/extended_euclidean.zig) | O(log n) |
| Extended Euclidean Algorithm (Python Filename Variant) | [`maths/extended_euclidean_algorithm.zig`](maths/extended_euclidean_algorithm.zig) | O(log n) |
| Modular Inverse | [`maths/modular_inverse.zig`](maths/modular_inverse.zig) | O(log m) |
| Euler's Totient | [`maths/eulers_totient.zig`](maths/eulers_totient.zig) | O(√n) |
| Chinese Remainder Theorem | [`maths/chinese_remainder_theorem.zig`](maths/chinese_remainder_theorem.zig) | O(k² + k log M) |
| Binomial Coefficient | [`maths/binomial_coefficient.zig`](maths/binomial_coefficient.zig) | O(min(k, n-k)) |
| Integer Square Root | [`maths/integer_square_root.zig`](maths/integer_square_root.zig) | O(log n) |
| Miller-Rabin Primality Test | [`maths/miller_rabin.zig`](maths/miller_rabin.zig) | O(k · log³ n), k=7 |
| Matrix Exponentiation | [`maths/matrix_exponentiation.zig`](maths/matrix_exponentiation.zig) | O(n³ log p) |
| Perfect Number Check | [`maths/perfect_number.zig`](maths/perfect_number.zig) | O(n) |
| Aliquot Sum | [`maths/aliquot_sum.zig`](maths/aliquot_sum.zig) | O(n) |
| Fermat Little Theorem (Mod Division) | [`maths/fermat_little_theorem.zig`](maths/fermat_little_theorem.zig) | O(log p) |
| Segmented Sieve | [`maths/segmented_sieve.zig`](maths/segmented_sieve.zig) | O(n log log n) |
| Odd Sieve | [`maths/odd_sieve.zig`](maths/odd_sieve.zig) | O(n log log n) |
| Twin Prime | [`maths/twin_prime.zig`](maths/twin_prime.zig) | O(√n) |
| Lucas Series | [`maths/lucas_series.zig`](maths/lucas_series.zig) | O(n) |
| Josephus Problem | [`maths/josephus_problem.zig`](maths/josephus_problem.zig) | O(n) |
| Sum of Digits | [`maths/sum_of_digits.zig`](maths/sum_of_digits.zig) | O(d) |
| Number of Digits | [`maths/number_of_digits.zig`](maths/number_of_digits.zig) | O(d) |
| Integer Palindrome Check | [`maths/is_int_palindrome.zig`](maths/is_int_palindrome.zig) | O(d) |
| Perfect Square | [`maths/perfect_square.zig`](maths/perfect_square.zig) | O(log n) |
| Perfect Cube | [`maths/perfect_cube.zig`](maths/perfect_cube.zig) | O(log n) |
| Quadratic Roots (Complex) | [`maths/quadratic_equations_complex_numbers.zig`](maths/quadratic_equations_complex_numbers.zig) | O(1) |
| Radix-2 FFT Polynomial Multiplication | [`maths/radix2_fft.zig`](maths/radix2_fft.zig) | O(n log n) |
| Decimal to Fraction | [`maths/decimal_to_fraction.zig`](maths/decimal_to_fraction.zig) | O(d) |
| Armstrong Numbers | [`maths/armstrong_numbers.zig`](maths/armstrong_numbers.zig) | O(d) |
| Automorphic Number | [`maths/automorphic_number.zig`](maths/automorphic_number.zig) | O(d) |
| Catalan Number | [`maths/catalan_number.zig`](maths/catalan_number.zig) | O(n) |
| Happy Number | [`maths/happy_number.zig`](maths/happy_number.zig) | O(k) |
| Hexagonal Number | [`maths/hexagonal_number.zig`](maths/hexagonal_number.zig) | O(1) |
| Pronic Number | [`maths/pronic_number.zig`](maths/pronic_number.zig) | O(log n) |
| Proth Number | [`maths/proth_number.zig`](maths/proth_number.zig) | O(n) |
| Triangular Numbers | [`maths/triangular_numbers.zig`](maths/triangular_numbers.zig) | O(1) |
| Hamming Numbers | [`maths/hamming_numbers.zig`](maths/hamming_numbers.zig) | O(n) |
| Polygonal Numbers | [`maths/polygonal_numbers.zig`](maths/polygonal_numbers.zig) | O(1) |
| Average Mean | [`maths/average_mean.zig`](maths/average_mean.zig) | O(n) |
| Average Median | [`maths/average_median.zig`](maths/average_median.zig) | O(n log n) |
| Average Mode | [`maths/average_mode.zig`](maths/average_mode.zig) | O(n) |
| Find Max | [`maths/find_max.zig`](maths/find_max.zig) | O(n) |
| Find Min | [`maths/find_min.zig`](maths/find_min.zig) | O(n) |
| Factors of a Number | [`maths/factors.zig`](maths/factors.zig) | O(sqrt(n)) |
| Geometric Mean | [`maths/geometric_mean.zig`](maths/geometric_mean.zig) | O(n) |
| Line Length (Arc Approximation) | [`maths/line_length.zig`](maths/line_length.zig) | O(steps) |
| Euclidean Distance | [`maths/euclidean_distance.zig`](maths/euclidean_distance.zig) | O(n) |
| Manhattan Distance | [`maths/manhattan_distance.zig`](maths/manhattan_distance.zig) | O(n) |
| Absolute Value Utilities | [`maths/abs.zig`](maths/abs.zig) | O(n) |
| Allocation Number | [`maths/allocation_number.zig`](maths/allocation_number.zig) | O(partitions) |
| Average Absolute Deviation | [`maths/average_absolute_deviation.zig`](maths/average_absolute_deviation.zig) | O(n) |
| Chebyshev Distance | [`maths/chebyshev_distance.zig`](maths/chebyshev_distance.zig) | O(n) |
| Minkowski Distance | [`maths/minkowski_distance.zig`](maths/minkowski_distance.zig) | O(n) |
| Jaccard Similarity | [`maths/jaccard_similarity.zig`](maths/jaccard_similarity.zig) | O(n + m) |
| Bailey-Borwein-Plouffe Pi Hex Digit Extraction | [`maths/bailey_borwein_plouffe.zig`](maths/bailey_borwein_plouffe.zig) | O((n + p) log n) |
| Decimal Isolate | [`maths/decimal_isolate.zig`](maths/decimal_isolate.zig) | O(1) |
| Floor Function | [`maths/floor.zig`](maths/floor.zig) | O(1) |
| Ceiling Function | [`maths/ceil.zig`](maths/ceil.zig) | O(1) |
| Signum Function | [`maths/signum.zig`](maths/signum.zig) | O(1) |
| Remove Digit for Maximum | [`maths/remove_digit.zig`](maths/remove_digit.zig) | O(d²) |
| Addition Without Arithmetic | [`maths/addition_without_arithmetic.zig`](maths/addition_without_arithmetic.zig) | O(1) |
| Arc Length | [`maths/arc_length.zig`](maths/arc_length.zig) | O(1) |
| Check Polygon Existence | [`maths/check_polygon.zig`](maths/check_polygon.zig) | O(n log n) |
| Chudnovsky π String Generator | [`maths/chudnovsky_algorithm.zig`](maths/chudnovsky_algorithm.zig) | O(p) digits via exact generator backend |
| Combinations (nCk) | [`maths/combinations.zig`](maths/combinations.zig) | O(k) |
| Double Factorial | [`maths/double_factorial.zig`](maths/double_factorial.zig) | O(n) |
| Dual Number Automatic Differentiation | [`maths/dual_number_automatic_differentiation.zig`](maths/dual_number_automatic_differentiation.zig) | O(p^2 · d) |
| Pythagoras 3D Distance | [`maths/pythagoras.zig`](maths/pythagoras.zig) | O(1) |
| Sum of Arithmetic Series | [`maths/sum_of_arithmetic_series.zig`](maths/sum_of_arithmetic_series.zig) | O(1) |
| Sum of Geometric Progression | [`maths/sum_of_geometric_progression.zig`](maths/sum_of_geometric_progression.zig) | O(log n) |
| Sum of Harmonic Progression | [`maths/sum_of_harmonic_series.zig`](maths/sum_of_harmonic_series.zig) | O(n) |
| Sylvester Sequence | [`maths/sylvester_sequence.zig`](maths/sylvester_sequence.zig) | O(n) |
| Two Sum (Hash Map) | [`maths/two_sum.zig`](maths/two_sum.zig) | O(n) |
| Two Pointer Two-Sum | [`maths/two_pointer.zig`](maths/two_pointer.zig) | O(n) |
| Three Sum | [`maths/three_sum.zig`](maths/three_sum.zig) | O(n²) |
| Triplet Sum | [`maths/triplet_sum.zig`](maths/triplet_sum.zig) | O(n²) |
| Sumset | [`maths/sumset.zig`](maths/sumset.zig) | O(n·m) |
| Max Sum Sliding Window | [`maths/max_sum_sliding_window.zig`](maths/max_sum_sliding_window.zig) | O(n) |
| Sock Merchant | [`maths/sock_merchant.zig`](maths/sock_merchant.zig) | O(n) |
| Polynomial Evaluation | [`maths/polynomial_evaluation.zig`](maths/polynomial_evaluation.zig) | O(n) |
| Kth Lexicographic Permutation | [`maths/kth_lexicographic_permutation.zig`](maths/kth_lexicographic_permutation.zig) | O(n²) |
| Largest of Very Large Numbers (Log Compare) | [`maths/largest_of_very_large_numbers.zig`](maths/largest_of_very_large_numbers.zig) | O(1) |
| Base -2 Conversion | [`maths/base_neg2_conversion.zig`](maths/base_neg2_conversion.zig) | O(log \|n\|) |
| Degrees to Radians | [`maths/radians.zig`](maths/radians.zig) | O(1) |
| Modular Exponential | [`maths/modular_exponential.zig`](maths/modular_exponential.zig) | O(log power) |
| Persistence (Multiplicative/Additive) | [`maths/persistence.zig`](maths/persistence.zig) | O(steps · digits) |
| IPv4 Address Validation | [`maths/is_ip_v4_address_valid.zig`](maths/is_ip_v4_address_valid.zig) | O(n) |
| Square-Free Factor List Check | [`maths/is_square_free.zig`](maths/is_square_free.zig) | O(n) |
| Juggler Sequence | [`maths/juggler_sequence.zig`](maths/juggler_sequence.zig) | O(sequence length) |
| Sophie Germain Prime Check | [`maths/germain_primes.zig`](maths/germain_primes.zig) | O(√n) |
| Greatest Common Divisor Variants | [`maths/greatest_common_divisor.zig`](maths/greatest_common_divisor.zig) | O(log n) |
| Lucas-Lehmer Primality Test | [`maths/lucas_lehmer_primality_test.zig`](maths/lucas_lehmer_primality_test.zig) | O(p) |
| GCD of N Numbers | [`maths/gcd_of_n_numbers.zig`](maths/gcd_of_n_numbers.zig) | O(n log m) |
| Prime Factors | [`maths/prime_factors.zig`](maths/prime_factors.zig) | O(√n) |
| Prime Numbers Generators | [`maths/prime_numbers.zig`](maths/prime_numbers.zig) | O(n√n) |
| Prime Sieve Eratosthenes | [`maths/prime_sieve_eratosthenes.zig`](maths/prime_sieve_eratosthenes.zig) | O(n log log n) |
| Pi Spigot Generator | [`maths/pi_generator.zig`](maths/pi_generator.zig) | superlinear big-int arithmetic |
| Power Using Recursion | [`maths/power_using_recursion.zig`](maths/power_using_recursion.zig) | O(exponent) |
| Pi Monte Carlo Estimation | [`maths/pi_monte_carlo_estimation.zig`](maths/pi_monte_carlo_estimation.zig) | O(simulations) |
| Liouville Lambda Function | [`maths/liouville_lambda.zig`](maths/liouville_lambda.zig) | O(√n) |
| Mobius Function | [`maths/mobius_function.zig`](maths/mobius_function.zig) | O(√n) |
| Monte Carlo Estimators | [`maths/monte_carlo.zig`](maths/monte_carlo.zig) | O(iterations) |
| Monte Carlo Dice Probability Estimation | [`maths/monte_carlo_dice.zig`](maths/monte_carlo_dice.zig) | O(throws · dice) |
| Interquartile Range | [`maths/interquartile_range.zig`](maths/interquartile_range.zig) | O(n log n) |
| Binary Exponentiation | [`maths/binary_exponentiation.zig`](maths/binary_exponentiation.zig) | O(log exponent) |
| Binary Multiplication | [`maths/binary_multiplication.zig`](maths/binary_multiplication.zig) | O(log b) |
| Area Under Curve | [`maths/area_under_curve.zig`](maths/area_under_curve.zig) | O(steps) |
| Area Formulas | [`maths/area.zig`](maths/area.zig) | O(1) |
| Trapezoidal Rule | [`maths/trapezoidal_rule.zig`](maths/trapezoidal_rule.zig) | O(steps) |
| Points Are Collinear in 3D | [`maths/points_are_collinear_3d.zig`](maths/points_are_collinear_3d.zig) | O(1) |
| Joint Probability Distribution | [`maths/joint_probability_distribution.zig`](maths/joint_probability_distribution.zig) | O(\|X\| · \|Y\|) |
| Fast Inverse Square Root | [`maths/fast_inverse_sqrt.zig`](maths/fast_inverse_sqrt.zig) | O(1) |
| Gaussian Function | [`maths/gaussian.zig`](maths/gaussian.zig) | O(1) |
| Gamma Function | [`maths/gamma.zig`](maths/gamma.zig) | Lanczos: O(1); Recursive: O(n) |
| Entropy (Information Theory) | [`maths/entropy.zig`](maths/entropy.zig) | O(n + alphabet^2) |
| Euler Method | [`maths/euler_method.zig`](maths/euler_method.zig) | O(steps) |
| Modified Euler Method | [`maths/euler_modified.zig`](maths/euler_modified.zig) | O(steps) |
| Hardy Ramanujan Distinct Prime Factors | [`maths/hardy_ramanujanalgo.zig`](maths/hardy_ramanujanalgo.zig) | O(sqrt(n)) |
| Pollard Rho Factorization | [`maths/pollard_rho.zig`](maths/pollard_rho.zig) | Probabilistic, expected sub-exponential |
| PrimeLib Utility Collection | [`maths/primelib.zig`](maths/primelib.zig) | Depends on subroutine |
| Print Multiplication Table | [`maths/print_multiplication_table.zig`](maths/print_multiplication_table.zig) | O(number_of_terms) |
| QR Decomposition (Householder) | [`maths/qr_decomposition.zig`](maths/qr_decomposition.zig) | O(m · n · min(m, n)) |
| Sin (Degrees Input) | [`maths/sin.zig`](maths/sin.zig) | O(accuracy) |
| Sigmoid Function | [`maths/sigmoid.zig`](maths/sigmoid.zig) | O(n) |
| Solovay-Strassen Primality Test | [`maths/solovay_strassen_primality_test.zig`](maths/solovay_strassen_primality_test.zig) | O(k · log^3 n) |
| Softmax Function | [`maths/softmax.zig`](maths/softmax.zig) | O(n) |
| Simultaneous Linear Equation Solver | [`maths/simultaneous_linear_equation_solver.zig`](maths/simultaneous_linear_equation_solver.zig) | O(n^3) |
| Hyperbolic Tangent | [`maths/tanh.zig`](maths/tanh.zig) | O(n) |
| Volume Formulas | [`maths/volume.zig`](maths/volume.zig) | O(1) |
| Modular Division | [`maths/modular_division.zig`](maths/modular_division.zig) | O(log n) |
| Maclaurin Series | [`maths/maclaurin_series.zig`](maths/maclaurin_series.zig) | O(k) |
| Dodecahedron Formulas | [`maths/dodecahedron.zig`](maths/dodecahedron.zig) | O(1) |
| Binomial Distribution | [`maths/binomial_distribution.zig`](maths/binomial_distribution.zig) | O(trials) |
| Basic Maths Utilities | [`maths/basic_maths.zig`](maths/basic_maths.zig) | O(√n) |
| Continued Fraction | [`maths/continued_fraction.zig`](maths/continued_fraction.zig) | O(k) |
| Karatsuba Multiplication | [`maths/karatsuba.zig`](maths/karatsuba.zig) | O(n^log2(3)) |
| Spearman Rank Correlation Coefficient | [`maths/spearman_rank_correlation_coefficient.zig`](maths/spearman_rank_correlation_coefficient.zig) | O(n log n) |
| Zeller's Congruence | [`maths/zellers_congruence.zig`](maths/zellers_congruence.zig) | O(1) |

### 数学 (144)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 最大公约数 | [`maths/gcd.zig`](maths/gcd.zig) | O(log n) |
| 最小公倍数 | [`maths/lcm.zig`](maths/lcm.zig) | O(log n) |
| 最小公倍数（Python 文件名变体） | [`maths/least_common_multiple.zig`](maths/least_common_multiple.zig) | 慢速：O(lcm / max(a, b))；快速：O(log n) |
| 斐波那契数列 | [`maths/fibonacci.zig`](maths/fibonacci.zig) | O(n) |
| 阶乘 | [`maths/factorial.zig`](maths/factorial.zig) | O(n) |
| 素数判定 | [`maths/prime_check.zig`](maths/prime_check.zig) | O(√n) |
| 埃拉托色尼筛法 | [`maths/sieve_of_eratosthenes.zig`](maths/sieve_of_eratosthenes.zig) | O(n log log n) |
| 快速幂 | [`maths/power.zig`](maths/power.zig) | O(log n) |
| 考拉兹序列 | [`maths/collatz_sequence.zig`](maths/collatz_sequence.zig) | O(?) |
| 扩展欧几里得算法 | [`maths/extended_euclidean.zig`](maths/extended_euclidean.zig) | O(log n) |
| 扩展欧几里得算法（Python 文件名变体） | [`maths/extended_euclidean_algorithm.zig`](maths/extended_euclidean_algorithm.zig) | O(log n) |
| 模逆 | [`maths/modular_inverse.zig`](maths/modular_inverse.zig) | O(log m) |
| 欧拉函数（Totient） | [`maths/eulers_totient.zig`](maths/eulers_totient.zig) | O(√n) |
| 中国剩余定理 | [`maths/chinese_remainder_theorem.zig`](maths/chinese_remainder_theorem.zig) | O(k² + k log M) |
| 二项式系数 | [`maths/binomial_coefficient.zig`](maths/binomial_coefficient.zig) | O(min(k, n-k)) |
| 整数平方根 | [`maths/integer_square_root.zig`](maths/integer_square_root.zig) | O(log n) |
| Miller-Rabin 素性测试 | [`maths/miller_rabin.zig`](maths/miller_rabin.zig) | O(k · log³ n)，k=7 |
| 矩阵快速幂 | [`maths/matrix_exponentiation.zig`](maths/matrix_exponentiation.zig) | O(n³ log p) |
| 完全数判定 | [`maths/perfect_number.zig`](maths/perfect_number.zig) | O(n) |
| Aliquot Sum（真因子和） | [`maths/aliquot_sum.zig`](maths/aliquot_sum.zig) | O(n) |
| 费马小定理（模除法） | [`maths/fermat_little_theorem.zig`](maths/fermat_little_theorem.zig) | O(log p) |
| 分段筛法 | [`maths/segmented_sieve.zig`](maths/segmented_sieve.zig) | O(n log log n) |
| 奇数筛法 | [`maths/odd_sieve.zig`](maths/odd_sieve.zig) | O(n log log n) |
| 孪生素数 | [`maths/twin_prime.zig`](maths/twin_prime.zig) | O(√n) |
| Lucas 数列 | [`maths/lucas_series.zig`](maths/lucas_series.zig) | O(n) |
| 约瑟夫问题 | [`maths/josephus_problem.zig`](maths/josephus_problem.zig) | O(n) |
| 各位数字之和 | [`maths/sum_of_digits.zig`](maths/sum_of_digits.zig) | O(d) |
| 数字位数统计 | [`maths/number_of_digits.zig`](maths/number_of_digits.zig) | O(d) |
| 整数回文判断 | [`maths/is_int_palindrome.zig`](maths/is_int_palindrome.zig) | O(d) |
| 完全平方数判断 | [`maths/perfect_square.zig`](maths/perfect_square.zig) | O(log n) |
| 完全立方数判断 | [`maths/perfect_cube.zig`](maths/perfect_cube.zig) | O(log n) |
| 一元二次方程复数根 | [`maths/quadratic_equations_complex_numbers.zig`](maths/quadratic_equations_complex_numbers.zig) | O(1) |
| Radix-2 FFT 多项式乘法 | [`maths/radix2_fft.zig`](maths/radix2_fft.zig) | O(n log n) |
| 小数转分数 | [`maths/decimal_to_fraction.zig`](maths/decimal_to_fraction.zig) | O(d) |
| 阿姆斯特朗数（自幂数） | [`maths/armstrong_numbers.zig`](maths/armstrong_numbers.zig) | O(d) |
| 自守数判断 | [`maths/automorphic_number.zig`](maths/automorphic_number.zig) | O(d) |
| Catalan 数（递推） | [`maths/catalan_number.zig`](maths/catalan_number.zig) | O(n) |
| 快乐数判断 | [`maths/happy_number.zig`](maths/happy_number.zig) | O(k) |
| 六边形数 | [`maths/hexagonal_number.zig`](maths/hexagonal_number.zig) | O(1) |
| Pronic 数判断 | [`maths/pronic_number.zig`](maths/pronic_number.zig) | O(log n) |
| Proth 数 | [`maths/proth_number.zig`](maths/proth_number.zig) | O(n) |
| 三角数 | [`maths/triangular_numbers.zig`](maths/triangular_numbers.zig) | O(1) |
| Hamming 数列 | [`maths/hamming_numbers.zig`](maths/hamming_numbers.zig) | O(n) |
| 多边形数 | [`maths/polygonal_numbers.zig`](maths/polygonal_numbers.zig) | O(1) |
| 算术平均值 | [`maths/average_mean.zig`](maths/average_mean.zig) | O(n) |
| 中位数 | [`maths/average_median.zig`](maths/average_median.zig) | O(n log n) |
| 众数 | [`maths/average_mode.zig`](maths/average_mode.zig) | O(n) |
| 最大值查找 | [`maths/find_max.zig`](maths/find_max.zig) | O(n) |
| 最小值查找 | [`maths/find_min.zig`](maths/find_min.zig) | O(n) |
| 因子分解（全部因子） | [`maths/factors.zig`](maths/factors.zig) | O(sqrt(n)) |
| 几何平均值 | [`maths/geometric_mean.zig`](maths/geometric_mean.zig) | O(n) |
| 曲线弧长近似 | [`maths/line_length.zig`](maths/line_length.zig) | O(steps) |
| 欧氏距离 | [`maths/euclidean_distance.zig`](maths/euclidean_distance.zig) | O(n) |
| 曼哈顿距离 | [`maths/manhattan_distance.zig`](maths/manhattan_distance.zig) | O(n) |
| 绝对值工具集 | [`maths/abs.zig`](maths/abs.zig) | O(n) |
| 字节分配区间 | [`maths/allocation_number.zig`](maths/allocation_number.zig) | O(partitions) |
| 平均绝对偏差 | [`maths/average_absolute_deviation.zig`](maths/average_absolute_deviation.zig) | O(n) |
| 切比雪夫距离 | [`maths/chebyshev_distance.zig`](maths/chebyshev_distance.zig) | O(n) |
| Minkowski 距离 | [`maths/minkowski_distance.zig`](maths/minkowski_distance.zig) | O(n) |
| Jaccard 相似度 | [`maths/jaccard_similarity.zig`](maths/jaccard_similarity.zig) | O(n + m) |
| Bailey-Borwein-Plouffe 圆周率十六进制位提取 | [`maths/bailey_borwein_plouffe.zig`](maths/bailey_borwein_plouffe.zig) | O((n + p) log n) |
| 小数部分提取 | [`maths/decimal_isolate.zig`](maths/decimal_isolate.zig) | O(1) |
| 向下取整 | [`maths/floor.zig`](maths/floor.zig) | O(1) |
| 向上取整 | [`maths/ceil.zig`](maths/ceil.zig) | O(1) |
| 符号函数 | [`maths/signum.zig`](maths/signum.zig) | O(1) |
| 删一位取最大值 | [`maths/remove_digit.zig`](maths/remove_digit.zig) | O(d²) |
| 位运算加法（无算术运算符） | [`maths/addition_without_arithmetic.zig`](maths/addition_without_arithmetic.zig) | O(1) |
| 弧长计算 | [`maths/arc_length.zig`](maths/arc_length.zig) | O(1) |
| 多边形可构成性检查 | [`maths/check_polygon.zig`](maths/check_polygon.zig) | O(n log n) |
| Chudnovsky π 字符串生成 | [`maths/chudnovsky_algorithm.zig`](maths/chudnovsky_algorithm.zig) | 通过精确位生成后端输出 O(p) 位 |
| 组合数（nCk） | [`maths/combinations.zig`](maths/combinations.zig) | O(k) |
| 双阶乘 | [`maths/double_factorial.zig`](maths/double_factorial.zig) | O(n) |
| 对偶数自动微分 | [`maths/dual_number_automatic_differentiation.zig`](maths/dual_number_automatic_differentiation.zig) | O(p^2 · d) |
| 勾股定理三维距离 | [`maths/pythagoras.zig`](maths/pythagoras.zig) | O(1) |
| 等差数列求和 | [`maths/sum_of_arithmetic_series.zig`](maths/sum_of_arithmetic_series.zig) | O(1) |
| 等比数列求和 | [`maths/sum_of_geometric_progression.zig`](maths/sum_of_geometric_progression.zig) | O(log n) |
| 调和级数求和 | [`maths/sum_of_harmonic_series.zig`](maths/sum_of_harmonic_series.zig) | O(n) |
| Sylvester 数列 | [`maths/sylvester_sequence.zig`](maths/sylvester_sequence.zig) | O(n) |
| 两数之和（哈希法） | [`maths/two_sum.zig`](maths/two_sum.zig) | O(n) |
| 双指针两数之和 | [`maths/two_pointer.zig`](maths/two_pointer.zig) | O(n) |
| 三数之和 | [`maths/three_sum.zig`](maths/three_sum.zig) | O(n²) |
| 三元组求和 | [`maths/triplet_sum.zig`](maths/triplet_sum.zig) | O(n²) |
| Sumset（集合和） | [`maths/sumset.zig`](maths/sumset.zig) | O(n·m) |
| 滑动窗口最大和 | [`maths/max_sum_sliding_window.zig`](maths/max_sum_sliding_window.zig) | O(n) |
| Sock Merchant（袜子配对） | [`maths/sock_merchant.zig`](maths/sock_merchant.zig) | O(n) |
| 多项式求值 | [`maths/polynomial_evaluation.zig`](maths/polynomial_evaluation.zig) | O(n) |
| 第 k 个字典序排列 | [`maths/kth_lexicographic_permutation.zig`](maths/kth_lexicographic_permutation.zig) | O(n²) |
| 超大幂比较（对数法） | [`maths/largest_of_very_large_numbers.zig`](maths/largest_of_very_large_numbers.zig) | O(1) |
| 负二进制转换 | [`maths/base_neg2_conversion.zig`](maths/base_neg2_conversion.zig) | O(log \|n\|) |
| 角度转弧度 | [`maths/radians.zig`](maths/radians.zig) | O(1) |
| 模幂运算 | [`maths/modular_exponential.zig`](maths/modular_exponential.zig) | O(log power) |
| 数位持久性（乘法/加法） | [`maths/persistence.zig`](maths/persistence.zig) | O(steps · digits) |
| IPv4 地址校验 | [`maths/is_ip_v4_address_valid.zig`](maths/is_ip_v4_address_valid.zig) | O(n) |
| 平方因子重复检查 | [`maths/is_square_free.zig`](maths/is_square_free.zig) | O(n) |
| Juggler 序列 | [`maths/juggler_sequence.zig`](maths/juggler_sequence.zig) | O(序列长度) |
| Sophie Germain 质数检查 | [`maths/germain_primes.zig`](maths/germain_primes.zig) | O(√n) |
| 最大公约数变体 | [`maths/greatest_common_divisor.zig`](maths/greatest_common_divisor.zig) | O(log n) |
| Lucas-Lehmer 素性测试 | [`maths/lucas_lehmer_primality_test.zig`](maths/lucas_lehmer_primality_test.zig) | O(p) |
| 多数最大公约数 | [`maths/gcd_of_n_numbers.zig`](maths/gcd_of_n_numbers.zig) | O(n log m) |
| 质因数分解 | [`maths/prime_factors.zig`](maths/prime_factors.zig) | O(√n) |
| 质数生成器 | [`maths/prime_numbers.zig`](maths/prime_numbers.zig) | O(n√n) |
| 埃拉托斯特尼筛变体 | [`maths/prime_sieve_eratosthenes.zig`](maths/prime_sieve_eratosthenes.zig) | O(n log log n) |
| π 位流生成器 | [`maths/pi_generator.zig`](maths/pi_generator.zig) | 超线性大整数运算 |
| 递归幂运算 | [`maths/power_using_recursion.zig`](maths/power_using_recursion.zig) | O(exponent) |
| Monte Carlo 圆周率估计 | [`maths/pi_monte_carlo_estimation.zig`](maths/pi_monte_carlo_estimation.zig) | O(simulations) |
| Liouville Lambda 函数 | [`maths/liouville_lambda.zig`](maths/liouville_lambda.zig) | O(√n) |
| Mobius 函数 | [`maths/mobius_function.zig`](maths/mobius_function.zig) | O(√n) |
| Monte Carlo 估计器 | [`maths/monte_carlo.zig`](maths/monte_carlo.zig) | O(iterations) |
| Monte Carlo 骰子概率估计 | [`maths/monte_carlo_dice.zig`](maths/monte_carlo_dice.zig) | O(throws · dice) |
| 四分位距 | [`maths/interquartile_range.zig`](maths/interquartile_range.zig) | O(n log n) |
| 二分幂 | [`maths/binary_exponentiation.zig`](maths/binary_exponentiation.zig) | O(log exponent) |
| 二进制乘法 | [`maths/binary_multiplication.zig`](maths/binary_multiplication.zig) | O(log b) |
| 曲线下面积 | [`maths/area_under_curve.zig`](maths/area_under_curve.zig) | O(steps) |
| 面积公式集 | [`maths/area.zig`](maths/area.zig) | O(1) |
| 梯形积分公式 | [`maths/trapezoidal_rule.zig`](maths/trapezoidal_rule.zig) | O(steps) |
| 三维点共线判断 | [`maths/points_are_collinear_3d.zig`](maths/points_are_collinear_3d.zig) | O(1) |
| 联合概率分布 | [`maths/joint_probability_distribution.zig`](maths/joint_probability_distribution.zig) | O(\|X\| · \|Y\|) |
| 快速平方根倒数 | [`maths/fast_inverse_sqrt.zig`](maths/fast_inverse_sqrt.zig) | O(1) |
| 高斯函数 | [`maths/gaussian.zig`](maths/gaussian.zig) | O(1) |
| Gamma 函数 | [`maths/gamma.zig`](maths/gamma.zig) | Lanczos：O(1)；递归：O(n) |
| 信息熵 | [`maths/entropy.zig`](maths/entropy.zig) | O(n + alphabet^2) |
| Euler 方法 | [`maths/euler_method.zig`](maths/euler_method.zig) | O(steps) |
| 改进 Euler 方法 | [`maths/euler_modified.zig`](maths/euler_modified.zig) | O(steps) |
| Hardy-Ramanujan 不同质因子计数 | [`maths/hardy_ramanujanalgo.zig`](maths/hardy_ramanujanalgo.zig) | O(sqrt(n)) |
| Pollard Rho 因子分解 | [`maths/pollard_rho.zig`](maths/pollard_rho.zig) | 概率型，期望次指数 |
| PrimeLib 聚合工具集 | [`maths/primelib.zig`](maths/primelib.zig) | 依具体子函数而定 |
| 乘法表输出 | [`maths/print_multiplication_table.zig`](maths/print_multiplication_table.zig) | O(number_of_terms) |
| QR 分解（Householder） | [`maths/qr_decomposition.zig`](maths/qr_decomposition.zig) | O(m · n · min(m, n)) |
| 正弦函数（角度输入） | [`maths/sin.zig`](maths/sin.zig) | O(accuracy) |
| Sigmoid 函数 | [`maths/sigmoid.zig`](maths/sigmoid.zig) | O(n) |
| Solovay-Strassen 素性测试 | [`maths/solovay_strassen_primality_test.zig`](maths/solovay_strassen_primality_test.zig) | O(k · log^3 n) |
| Softmax 函数 | [`maths/softmax.zig`](maths/softmax.zig) | O(n) |
| 线性方程组求解器 | [`maths/simultaneous_linear_equation_solver.zig`](maths/simultaneous_linear_equation_solver.zig) | O(n^3) |
| 双曲正切 | [`maths/tanh.zig`](maths/tanh.zig) | O(n) |
| 体积公式集 | [`maths/volume.zig`](maths/volume.zig) | O(1) |
| 模除法 | [`maths/modular_division.zig`](maths/modular_division.zig) | O(log n) |
| Maclaurin 级数 | [`maths/maclaurin_series.zig`](maths/maclaurin_series.zig) | O(k) |
| 正十二面体公式 | [`maths/dodecahedron.zig`](maths/dodecahedron.zig) | O(1) |
| 二项分布 | [`maths/binomial_distribution.zig`](maths/binomial_distribution.zig) | O(trials) |
| 基础数学工具 | [`maths/basic_maths.zig`](maths/basic_maths.zig) | O(√n) |
| 连分数 | [`maths/continued_fraction.zig`](maths/continued_fraction.zig) | O(k) |
| Karatsuba 乘法 | [`maths/karatsuba.zig`](maths/karatsuba.zig) | O(n^log2(3)) |
| Spearman 秩相关系数 | [`maths/spearman_rank_correlation_coefficient.zig`](maths/spearman_rank_correlation_coefficient.zig) | O(n log n) |
| Zeller 同余公式 | [`maths/zellers_congruence.zig`](maths/zellers_congruence.zig) | O(1) |

### Bit Manipulation (27)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Is Power of Two | [`bit_manipulation/is_power_of_two.zig`](bit_manipulation/is_power_of_two.zig) | O(1) |
| Count Set Bits | [`bit_manipulation/count_set_bits.zig`](bit_manipulation/count_set_bits.zig) | O(k) |
| Find Unique Number | [`bit_manipulation/find_unique_number.zig`](bit_manipulation/find_unique_number.zig) | O(n) |
| Reverse Bits | [`bit_manipulation/reverse_bits.zig`](bit_manipulation/reverse_bits.zig) | O(1) |
| Missing Number | [`bit_manipulation/missing_number.zig`](bit_manipulation/missing_number.zig) | O(n) |
| Is Power of Four | [`bit_manipulation/power_of_4.zig`](bit_manipulation/power_of_4.zig) | O(1) |
| Gray Code Sequence | [`bit_manipulation/gray_code_sequence.zig`](bit_manipulation/gray_code_sequence.zig) | O(2^n) |
| Highest Set Bit Position | [`bit_manipulation/highest_set_bit.zig`](bit_manipulation/highest_set_bit.zig) | O(log n) |
| Index of Rightmost Set Bit | [`bit_manipulation/index_of_rightmost_set_bit.zig`](bit_manipulation/index_of_rightmost_set_bit.zig) | O(log n) |
| Find Previous Power of Two | [`bit_manipulation/find_previous_power_of_two.zig`](bit_manipulation/find_previous_power_of_two.zig) | O(log n) |
| Swap Odd and Even Bits | [`bit_manipulation/swap_all_odd_and_even_bits.zig`](bit_manipulation/swap_all_odd_and_even_bits.zig) | O(1) |
| Different Signs Check | [`bit_manipulation/numbers_different_signs.zig`](bit_manipulation/numbers_different_signs.zig) | O(1) |
| Is Even | [`bit_manipulation/is_even.zig`](bit_manipulation/is_even.zig) | O(1) |
| Binary Count Trailing Zeros | [`bit_manipulation/binary_count_trailing_zeros.zig`](bit_manipulation/binary_count_trailing_zeros.zig) | O(k) |
| Bitwise Addition (Recursive) | [`bit_manipulation/bitwise_addition_recursive.zig`](bit_manipulation/bitwise_addition_recursive.zig) | O(w) |
| Binary AND Operator | [`bit_manipulation/binary_and_operator.zig`](bit_manipulation/binary_and_operator.zig) | O(w) |
| Binary OR Operator | [`bit_manipulation/binary_or_operator.zig`](bit_manipulation/binary_or_operator.zig) | O(w) |
| Binary XOR Operator | [`bit_manipulation/binary_xor_operator.zig`](bit_manipulation/binary_xor_operator.zig) | O(w) |
| Binary Shifts | [`bit_manipulation/binary_shifts.zig`](bit_manipulation/binary_shifts.zig) | O(w + s) |
| Binary Two's Complement | [`bit_manipulation/binary_twos_complement.zig`](bit_manipulation/binary_twos_complement.zig) | O(w) |
| Single Bit Manipulation Operations | [`bit_manipulation/single_bit_manipulation_operations.zig`](bit_manipulation/single_bit_manipulation_operations.zig) | O(1) |
| Binary Coded Decimal | [`bit_manipulation/binary_coded_decimal.zig`](bit_manipulation/binary_coded_decimal.zig) | O(d) |
| Excess-3 Code | [`bit_manipulation/excess_3_code.zig`](bit_manipulation/excess_3_code.zig) | O(d) |
| Binary Count Set Bits | [`bit_manipulation/binary_count_setbits.zig`](bit_manipulation/binary_count_setbits.zig) | O(w) |
| Count 1s (Brian Kernighan) | [`bit_manipulation/count_1s_brian_kernighan_method.zig`](bit_manipulation/count_1s_brian_kernighan_method.zig) | O(k) |
| Count Number of One Bits | [`bit_manipulation/count_number_of_one_bits.zig`](bit_manipulation/count_number_of_one_bits.zig) | O(k) / O(w) |
| Largest Power of Two <= Number | [`bit_manipulation/largest_pow_of_two_le_num.zig`](bit_manipulation/largest_pow_of_two_le_num.zig) | O(log n) |

### 位运算 (27)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 判断 2 的幂 | [`bit_manipulation/is_power_of_two.zig`](bit_manipulation/is_power_of_two.zig) | O(1) |
| 统计置位数 | [`bit_manipulation/count_set_bits.zig`](bit_manipulation/count_set_bits.zig) | O(k) |
| 找唯一数 | [`bit_manipulation/find_unique_number.zig`](bit_manipulation/find_unique_number.zig) | O(n) |
| 位翻转 | [`bit_manipulation/reverse_bits.zig`](bit_manipulation/reverse_bits.zig) | O(1) |
| 缺失数字 | [`bit_manipulation/missing_number.zig`](bit_manipulation/missing_number.zig) | O(n) |
| 判断 4 的幂 | [`bit_manipulation/power_of_4.zig`](bit_manipulation/power_of_4.zig) | O(1) |
| 格雷码序列 | [`bit_manipulation/gray_code_sequence.zig`](bit_manipulation/gray_code_sequence.zig) | O(2^n) |
| 最高置位位置 | [`bit_manipulation/highest_set_bit.zig`](bit_manipulation/highest_set_bit.zig) | O(log n) |
| 最右置位索引 | [`bit_manipulation/index_of_rightmost_set_bit.zig`](bit_manipulation/index_of_rightmost_set_bit.zig) | O(log n) |
| 不超过 n 的最大 2 的幂 | [`bit_manipulation/find_previous_power_of_two.zig`](bit_manipulation/find_previous_power_of_two.zig) | O(log n) |
| 奇偶位交换 | [`bit_manipulation/swap_all_odd_and_even_bits.zig`](bit_manipulation/swap_all_odd_and_even_bits.zig) | O(1) |
| 异号判断 | [`bit_manipulation/numbers_different_signs.zig`](bit_manipulation/numbers_different_signs.zig) | O(1) |
| 偶数判断 | [`bit_manipulation/is_even.zig`](bit_manipulation/is_even.zig) | O(1) |
| 二进制末尾零计数 | [`bit_manipulation/binary_count_trailing_zeros.zig`](bit_manipulation/binary_count_trailing_zeros.zig) | O(k) |
| 递归位运算加法 | [`bit_manipulation/bitwise_addition_recursive.zig`](bit_manipulation/bitwise_addition_recursive.zig) | O(w) |
| 二进制 AND 运算 | [`bit_manipulation/binary_and_operator.zig`](bit_manipulation/binary_and_operator.zig) | O(w) |
| 二进制 OR 运算 | [`bit_manipulation/binary_or_operator.zig`](bit_manipulation/binary_or_operator.zig) | O(w) |
| 二进制 XOR 运算 | [`bit_manipulation/binary_xor_operator.zig`](bit_manipulation/binary_xor_operator.zig) | O(w) |
| 二进制移位 | [`bit_manipulation/binary_shifts.zig`](bit_manipulation/binary_shifts.zig) | O(w + s) |
| 二进制补码表示 | [`bit_manipulation/binary_twos_complement.zig`](bit_manipulation/binary_twos_complement.zig) | O(w) |
| 单比特操作集合 | [`bit_manipulation/single_bit_manipulation_operations.zig`](bit_manipulation/single_bit_manipulation_operations.zig) | O(1) |
| 二进制编码十进制（BCD） | [`bit_manipulation/binary_coded_decimal.zig`](bit_manipulation/binary_coded_decimal.zig) | O(d) |
| Excess-3 码 | [`bit_manipulation/excess_3_code.zig`](bit_manipulation/excess_3_code.zig) | O(d) |
| 二进制置位计数 | [`bit_manipulation/binary_count_setbits.zig`](bit_manipulation/binary_count_setbits.zig) | O(w) |
| Brian Kernighan 置位计数 | [`bit_manipulation/count_1s_brian_kernighan_method.zig`](bit_manipulation/count_1s_brian_kernighan_method.zig) | O(k) |
| 1 比特数量统计 | [`bit_manipulation/count_number_of_one_bits.zig`](bit_manipulation/count_number_of_one_bits.zig) | O(k) / O(w) |
| 不超过 n 的最大 2 的幂（别名实现） | [`bit_manipulation/largest_pow_of_two_le_num.zig`](bit_manipulation/largest_pow_of_two_le_num.zig) | O(log n) |

### Conversions (27)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Decimal to Binary | [`conversions/decimal_to_binary.zig`](conversions/decimal_to_binary.zig) | O(log n) |
| Binary to Decimal | [`conversions/binary_to_decimal.zig`](conversions/binary_to_decimal.zig) | O(n) |
| Decimal to Hexadecimal | [`conversions/decimal_to_hexadecimal.zig`](conversions/decimal_to_hexadecimal.zig) | O(log n) |
| Binary to Hexadecimal | [`conversions/binary_to_hexadecimal.zig`](conversions/binary_to_hexadecimal.zig) | O(n) |
| Roman to Integer | [`conversions/roman_to_integer.zig`](conversions/roman_to_integer.zig) | O(n) |
| Integer to Roman | [`conversions/integer_to_roman.zig`](conversions/integer_to_roman.zig) | O(1) bounded range |
| Temperature Conversion | [`conversions/temperature_conversion.zig`](conversions/temperature_conversion.zig) | O(1) |
| Octal to Binary | [`conversions/octal_to_binary.zig`](conversions/octal_to_binary.zig) | O(n) |
| Binary to Octal | [`conversions/binary_to_octal.zig`](conversions/binary_to_octal.zig) | O(n) |
| Octal to Decimal | [`conversions/octal_to_decimal.zig`](conversions/octal_to_decimal.zig) | O(n) |
| Octal to Hexadecimal | [`conversions/octal_to_hexadecimal.zig`](conversions/octal_to_hexadecimal.zig) | O(n) |
| Decimal to Octal | [`conversions/decimal_to_octal.zig`](conversions/decimal_to_octal.zig) | O(log n) |
| Hexadecimal to Decimal | [`conversions/hexadecimal_to_decimal.zig`](conversions/hexadecimal_to_decimal.zig) | O(n) |
| Excel Title to Column | [`conversions/excel_title_to_column.zig`](conversions/excel_title_to_column.zig) | O(n) |
| Decimal to Any Base | [`conversions/decimal_to_any.zig`](conversions/decimal_to_any.zig) | O(log_base(n)) |
| IPv4 <-> Decimal Conversion | [`conversions/ipv4_conversion.zig`](conversions/ipv4_conversion.zig) | O(n) |
| Hex to Binary Integer | [`conversions/hex_to_bin.zig`](conversions/hex_to_bin.zig) | O(n) |
| SI/Binary Prefix Conversions | [`conversions/prefix_conversions.zig`](conversions/prefix_conversions.zig) | O(1) |
| Length Conversion | [`conversions/length_conversion.zig`](conversions/length_conversion.zig) | O(1) |
| Speed Conversions | [`conversions/speed_conversions.zig`](conversions/speed_conversions.zig) | O(1) |
| Time Conversions | [`conversions/time_conversions.zig`](conversions/time_conversions.zig) | O(1) |
| Pressure Conversions | [`conversions/pressure_conversions.zig`](conversions/pressure_conversions.zig) | O(1) |
| Volume Conversions | [`conversions/volume_conversions.zig`](conversions/volume_conversions.zig) | O(1) |
| Energy Conversions | [`conversions/energy_conversions.zig`](conversions/energy_conversions.zig) | O(1) |
| Molecular Chemistry Utilities | [`conversions/molecular_chemistry.zig`](conversions/molecular_chemistry.zig) | O(1) |
| Rectangular to Polar Conversion | [`conversions/rectangular_to_polar.zig`](conversions/rectangular_to_polar.zig) | O(1) |
| SI/Binary Prefix String Conversion | [`conversions/prefix_conversions_string.zig`](conversions/prefix_conversions_string.zig) | O(1) |

### 进制转换 (27)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 十进制转二进制 | [`conversions/decimal_to_binary.zig`](conversions/decimal_to_binary.zig) | O(log n) |
| 二进制转十进制 | [`conversions/binary_to_decimal.zig`](conversions/binary_to_decimal.zig) | O(n) |
| 十进制转十六进制 | [`conversions/decimal_to_hexadecimal.zig`](conversions/decimal_to_hexadecimal.zig) | O(log n) |
| 二进制转十六进制 | [`conversions/binary_to_hexadecimal.zig`](conversions/binary_to_hexadecimal.zig) | O(n) |
| 罗马数字转整数 | [`conversions/roman_to_integer.zig`](conversions/roman_to_integer.zig) | O(n) |
| 整数转罗马数字 | [`conversions/integer_to_roman.zig`](conversions/integer_to_roman.zig) | O(1)（有界区间） |
| 温度单位转换 | [`conversions/temperature_conversion.zig`](conversions/temperature_conversion.zig) | O(1) |
| 八进制转二进制 | [`conversions/octal_to_binary.zig`](conversions/octal_to_binary.zig) | O(n) |
| 二进制转八进制 | [`conversions/binary_to_octal.zig`](conversions/binary_to_octal.zig) | O(n) |
| 八进制转十进制 | [`conversions/octal_to_decimal.zig`](conversions/octal_to_decimal.zig) | O(n) |
| 八进制转十六进制 | [`conversions/octal_to_hexadecimal.zig`](conversions/octal_to_hexadecimal.zig) | O(n) |
| 十进制转八进制 | [`conversions/decimal_to_octal.zig`](conversions/decimal_to_octal.zig) | O(log n) |
| 十六进制转十进制 | [`conversions/hexadecimal_to_decimal.zig`](conversions/hexadecimal_to_decimal.zig) | O(n) |
| Excel 列名转列号 | [`conversions/excel_title_to_column.zig`](conversions/excel_title_to_column.zig) | O(n) |
| 十进制转任意进制 | [`conversions/decimal_to_any.zig`](conversions/decimal_to_any.zig) | O(log_base(n)) |
| IPv4 与十进制互转 | [`conversions/ipv4_conversion.zig`](conversions/ipv4_conversion.zig) | O(n) |
| 十六进制转二进制整数 | [`conversions/hex_to_bin.zig`](conversions/hex_to_bin.zig) | O(n) |
| SI/二进制前缀换算 | [`conversions/prefix_conversions.zig`](conversions/prefix_conversions.zig) | O(1) |
| 长度单位换算 | [`conversions/length_conversion.zig`](conversions/length_conversion.zig) | O(1) |
| 速度单位换算 | [`conversions/speed_conversions.zig`](conversions/speed_conversions.zig) | O(1) |
| 时间单位换算 | [`conversions/time_conversions.zig`](conversions/time_conversions.zig) | O(1) |
| 压力单位换算 | [`conversions/pressure_conversions.zig`](conversions/pressure_conversions.zig) | O(1) |
| 体积单位换算 | [`conversions/volume_conversions.zig`](conversions/volume_conversions.zig) | O(1) |
| 能量单位换算 | [`conversions/energy_conversions.zig`](conversions/energy_conversions.zig) | O(1) |
| 分子化学计算工具 | [`conversions/molecular_chemistry.zig`](conversions/molecular_chemistry.zig) | O(1) |
| 直角坐标转极坐标 | [`conversions/rectangular_to_polar.zig`](conversions/rectangular_to_polar.zig) | O(1) |
| SI/二进制前缀字符串转换 | [`conversions/prefix_conversions_string.zig`](conversions/prefix_conversions_string.zig) | O(1) |
