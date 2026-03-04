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

### Searching (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Linear Search | [`searches/linear_search.zig`](searches/linear_search.zig) | O(n) |
| Binary Search | [`searches/binary_search.zig`](searches/binary_search.zig) | O(log n) |
| Exponential Search | [`searches/exponential_search.zig`](searches/exponential_search.zig) | O(log n) |
| Interpolation Search | [`searches/interpolation_search.zig`](searches/interpolation_search.zig) | O(log log n) avg |
| Jump Search | [`searches/jump_search.zig`](searches/jump_search.zig) | O(√n) |
| Ternary Search | [`searches/ternary_search.zig`](searches/ternary_search.zig) | O(log₃ n) |

### Math (81)

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
| Average Absolute Deviation | [`maths/average_absolute_deviation.zig`](maths/average_absolute_deviation.zig) | O(n) |
| Chebyshev Distance | [`maths/chebyshev_distance.zig`](maths/chebyshev_distance.zig) | O(n) |
| Minkowski Distance | [`maths/minkowski_distance.zig`](maths/minkowski_distance.zig) | O(n) |
| Jaccard Similarity | [`maths/jaccard_similarity.zig`](maths/jaccard_similarity.zig) | O(n + m) |
| Decimal Isolate | [`maths/decimal_isolate.zig`](maths/decimal_isolate.zig) | O(1) |
| Floor Function | [`maths/floor.zig`](maths/floor.zig) | O(1) |
| Ceiling Function | [`maths/ceil.zig`](maths/ceil.zig) | O(1) |
| Signum Function | [`maths/signum.zig`](maths/signum.zig) | O(1) |
| Remove Digit for Maximum | [`maths/remove_digit.zig`](maths/remove_digit.zig) | O(d²) |
| Addition Without Arithmetic | [`maths/addition_without_arithmetic.zig`](maths/addition_without_arithmetic.zig) | O(1) |
| Arc Length | [`maths/arc_length.zig`](maths/arc_length.zig) | O(1) |
| Check Polygon Existence | [`maths/check_polygon.zig`](maths/check_polygon.zig) | O(n log n) |
| Combinations (nCk) | [`maths/combinations.zig`](maths/combinations.zig) | O(k) |
| Double Factorial | [`maths/double_factorial.zig`](maths/double_factorial.zig) | O(n) |
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

### Graphs (46)

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
| Articulation Points | [`graphs/articulation_points.zig`](graphs/articulation_points.zig) | O(V + E) |
| Kosaraju SCC | [`graphs/kosaraju_scc.zig`](graphs/kosaraju_scc.zig) | O(V + E) |
| Kahn Topological Sort | [`graphs/kahn_topological_sort.zig`](graphs/kahn_topological_sort.zig) | O(V + E) |
| BFS Shortest Path | [`graphs/breadth_first_search_shortest_path.zig`](graphs/breadth_first_search_shortest_path.zig) | O(V + E) |
| Boruvka MST | [`graphs/boruvka_mst.zig`](graphs/boruvka_mst.zig) | O(E log V) |
| 0-1 BFS Shortest Path | [`graphs/zero_one_bfs_shortest_path.zig`](graphs/zero_one_bfs_shortest_path.zig) | O(V + E) |
| Bidirectional BFS Path | [`graphs/bidirectional_breadth_first_search.zig`](graphs/bidirectional_breadth_first_search.zig) | O(V + E) |
| Dijkstra on Binary Grid | [`graphs/dijkstra_binary_grid.zig`](graphs/dijkstra_binary_grid.zig) | O((R·C)²) |
| Even Tree | [`graphs/even_tree.zig`](graphs/even_tree.zig) | O(V + E) |
| Gale-Shapley Stable Matching | [`graphs/gale_shapley_stable_matching.zig`](graphs/gale_shapley_stable_matching.zig) | O(n²) |
| PageRank (Iterative) | [`graphs/page_rank.zig`](graphs/page_rank.zig) | O(iterations·(V + E)) |
| Bidirectional Dijkstra | [`graphs/bidirectional_dijkstra.zig`](graphs/bidirectional_dijkstra.zig) | O(V² + E) |
| Greedy Best-First Search | [`graphs/greedy_best_first.zig`](graphs/greedy_best_first.zig) | O(V²) |
| Dinic Max Flow | [`graphs/dinic_max_flow.zig`](graphs/dinic_max_flow.zig) | O(V²·E) |
| Bidirectional Search | [`graphs/bidirectional_search.zig`](graphs/bidirectional_search.zig) | O(V + E) |
| Minimum Path Sum | [`graphs/minimum_path_sum.zig`](graphs/minimum_path_sum.zig) | O(rows·cols) |
| Deep Clone Graph | [`graphs/deep_clone_graph.zig`](graphs/deep_clone_graph.zig) | O(V + E) |
| Dijkstra (Matrix) | [`graphs/dijkstra_matrix.zig`](graphs/dijkstra_matrix.zig) | O(V²) |
| Breadth-First Search (Queue/Deque Variant) | [`graphs/breadth_first_search_2.zig`](graphs/breadth_first_search_2.zig) | O(V + E) |
| Depth-First Search (Iterative Variant) | [`graphs/depth_first_search_2.zig`](graphs/depth_first_search_2.zig) | O(V + E) |
| Dijkstra (Matrix Float Variant) | [`graphs/dijkstra_2.zig`](graphs/dijkstra_2.zig) | O(V²) |
| Dijkstra (Alternate Matrix Variant) | [`graphs/dijkstra_alternate.zig`](graphs/dijkstra_alternate.zig) | O(V²) |
| Greedy Minimum Vertex Cover (Approx.) | [`graphs/greedy_min_vertex_cover.zig`](graphs/greedy_min_vertex_cover.zig) | O(V² + V·E) |
| Matching Minimum Vertex Cover (Approx.) | [`graphs/matching_min_vertex_cover.zig`](graphs/matching_min_vertex_cover.zig) | O(V³) worst |
| Karger Minimum Cut | [`graphs/karger_min_cut.zig`](graphs/karger_min_cut.zig) | O(trials·V·E) |
| Random Graph Generator | [`graphs/random_graph_generator.zig`](graphs/random_graph_generator.zig) | O(V²) |
| Markov Chain Transition Simulation | [`graphs/markov_chain.zig`](graphs/markov_chain.zig) | O(steps·N) |
| Kahn Longest Distance in DAG | [`graphs/kahn_longest_distance.zig`](graphs/kahn_longest_distance.zig) | O(V + E) |
| Graph Adjacency List Data Structure | [`graphs/graph_adjacency_list.zig`](graphs/graph_adjacency_list.zig) | O(1) avg edge insert/query, O(deg) removal |
| Graph Adjacency Matrix Data Structure | [`graphs/graph_adjacency_matrix.zig`](graphs/graph_adjacency_matrix.zig) | O(1) edge query/update, O(V²) vertex resize |

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

### Ciphers (47)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Caesar Cipher | [`ciphers/caesar_cipher.zig`](ciphers/caesar_cipher.zig) | O(n · m) |
| ROT13 / Caesar Shift | [`ciphers/rot13.zig`](ciphers/rot13.zig) | O(n) |
| Atbash Cipher | [`ciphers/atbash.zig`](ciphers/atbash.zig) | O(n) |
| Vigenere Cipher | [`ciphers/vigenere_cipher.zig`](ciphers/vigenere_cipher.zig) | O(n) |
| Rail Fence Cipher | [`ciphers/rail_fence_cipher.zig`](ciphers/rail_fence_cipher.zig) | O(n) |
| XOR Cipher | [`ciphers/xor_cipher.zig`](ciphers/xor_cipher.zig) | O(n) |
| Base64 Cipher | [`ciphers/base64_cipher.zig`](ciphers/base64_cipher.zig) | O(n) |
| Transposition Cipher (Route) | [`ciphers/transposition_cipher.zig`](ciphers/transposition_cipher.zig) | O(n) |
| A1Z26 Letter-Number Cipher | [`ciphers/a1z26.zig`](ciphers/a1z26.zig) | O(n) |
| Affine Cipher | [`ciphers/affine_cipher.zig`](ciphers/affine_cipher.zig) | O(n · m) |
| Baconian Cipher | [`ciphers/baconian_cipher.zig`](ciphers/baconian_cipher.zig) | O(n) |
| Base16 Encoding/Decoding | [`ciphers/base16.zig`](ciphers/base16.zig) | O(n) |
| Base32 Encoding/Decoding | [`ciphers/base32.zig`](ciphers/base32.zig) | O(n) |
| Base85 Encoding/Decoding | [`ciphers/base85.zig`](ciphers/base85.zig) | O(n) |
| Morse Code Cipher | [`ciphers/morse_code.zig`](ciphers/morse_code.zig) | O(n · table) |
| Polybius Square Cipher | [`ciphers/polybius.zig`](ciphers/polybius.zig) | O(n) |
| Autokey Cipher | [`ciphers/autokey.zig`](ciphers/autokey.zig) | O(n) |
| Beaufort Cipher | [`ciphers/beaufort_cipher.zig`](ciphers/beaufort_cipher.zig) | O(n) |
| Gronsfeld Cipher | [`ciphers/gronsfeld_cipher.zig`](ciphers/gronsfeld_cipher.zig) | O(n) |
| Vernam Cipher | [`ciphers/vernam_cipher.zig`](ciphers/vernam_cipher.zig) | O(n) |
| Running Key Cipher | [`ciphers/running_key_cipher.zig`](ciphers/running_key_cipher.zig) | O(n) |
| Onepad Cipher | [`ciphers/onepad_cipher.zig`](ciphers/onepad_cipher.zig) | O(n) |
| Permutation Cipher | [`ciphers/permutation_cipher.zig`](ciphers/permutation_cipher.zig) | O(n) |
| Mono Alphabetic Cipher | [`ciphers/mono_alphabetic_ciphers.zig`](ciphers/mono_alphabetic_ciphers.zig) | O(n) |
| Brute Force Caesar Cipher | [`ciphers/brute_force_caesar_cipher.zig`](ciphers/brute_force_caesar_cipher.zig) | O(26 · n) |
| Cryptomath Modular Inverse | [`ciphers/cryptomath_module.zig`](ciphers/cryptomath_module.zig) | O(log m) |
| Diffie Primitive Root Search | [`ciphers/diffie.zig`](ciphers/diffie.zig) | O(m² log m) |
| Deterministic Miller-Rabin | [`ciphers/deterministic_miller_rabin.zig`](ciphers/deterministic_miller_rabin.zig) | O(k · log³ n) |
| RSA Factorization (d,e,n) | [`ciphers/rsa_factorization.zig`](ciphers/rsa_factorization.zig) | randomized |
| Porta Cipher | [`ciphers/porta_cipher.zig`](ciphers/porta_cipher.zig) | O(n) |
| Mixed Keyword Cipher | [`ciphers/mixed_keyword_cypher.zig`](ciphers/mixed_keyword_cypher.zig) | O(n) |
| Simple Keyword Cipher | [`ciphers/simple_keyword_cypher.zig`](ciphers/simple_keyword_cypher.zig) | O(n) |
| Simple Substitution Cipher | [`ciphers/simple_substitution_cipher.zig`](ciphers/simple_substitution_cipher.zig) | O(n) |
| Rabin-Miller Primality Test | [`ciphers/rabin_miller.zig`](ciphers/rabin_miller.zig) | O(k · log³ n) |
| RSA Key Generator | [`ciphers/rsa_key_generator.zig`](ciphers/rsa_key_generator.zig) | probabilistic |
| RSA Cipher | [`ciphers/rsa_cipher.zig`](ciphers/rsa_cipher.zig) | O(blocks · log exp) |
| ElGamal Key Generator | [`ciphers/elgamal_key_generator.zig`](ciphers/elgamal_key_generator.zig) | probabilistic |
| Transposition File Wrapper | [`ciphers/transposition_cipher_encrypt_decrypt_file.zig`](ciphers/transposition_cipher_encrypt_decrypt_file.zig) | O(n) |
| Bifid Cipher | [`ciphers/bifid.zig`](ciphers/bifid.zig) | O(n) |
| Playfair Cipher | [`ciphers/playfair_cipher.zig`](ciphers/playfair_cipher.zig) | O(n) |
| Caesar Chi-Squared Decryption | [`ciphers/decrypt_caesar_with_chi_squared.zig`](ciphers/decrypt_caesar_with_chi_squared.zig) | O(26 · n²) |
| Fractionated Morse Cipher | [`ciphers/fractionated_morse_cipher.zig`](ciphers/fractionated_morse_cipher.zig) | O(n) |
| Hill Cipher | [`ciphers/hill_cipher.zig`](ciphers/hill_cipher.zig) | O(n) |
| Shuffled Shift Cipher | [`ciphers/shuffled_shift_cipher.zig`](ciphers/shuffled_shift_cipher.zig) | O(n · m) |
| Trifid Cipher | [`ciphers/trifid_cipher.zig`](ciphers/trifid_cipher.zig) | O(n) |
| Enigma Machine 2 | [`ciphers/enigma_machine2.zig`](ciphers/enigma_machine2.zig) | O(n · 26) |
| Diffie-Hellman Key Exchange | [`ciphers/diffie_hellman.zig`](ciphers/diffie_hellman.zig) | O(log exp) per pow |

### Hashing (1)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| SHA-256 | [`hashing/sha256.zig`](hashing/sha256.zig) | O(n) |

### Strings (38)

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
| Camel Case to Snake Case | [`strings/camel_case_to_snake_case.zig`](strings/camel_case_to_snake_case.zig) | O(n) |
| Palindrome Rearrangement Check | [`strings/can_string_be_rearranged_as_palindrome.zig`](strings/can_string_be_rearranged_as_palindrome.zig) | O(n) |
| Capitalize | [`strings/capitalize.zig`](strings/capitalize.zig) | O(n) |
| Count Vowels | [`strings/count_vowels.zig`](strings/count_vowels.zig) | O(n) |
| Contains Unique Characters | [`strings/is_contains_unique_chars.zig`](strings/is_contains_unique_chars.zig) | O(n) |
| Is Isogram | [`strings/is_isogram.zig`](strings/is_isogram.zig) | O(n) |
| Join Strings | [`strings/join.zig`](strings/join.zig) | O(total_len) |
| Lowercase ASCII | [`strings/lower.zig`](strings/lower.zig) | O(n) |
| Split String | [`strings/split.zig`](strings/split.zig) | O(n) |
| Uppercase ASCII | [`strings/upper.zig`](strings/upper.zig) | O(n) |
| Alternative String Arrange | [`strings/alternative_string_arrange.zig`](strings/alternative_string_arrange.zig) | O(n + m) |
| Boyer-Moore Search | [`strings/boyer_moore_search.zig`](strings/boyer_moore_search.zig) | O(n·m) worst |
| Bitap String Match | [`strings/bitap_string_match.zig`](strings/bitap_string_match.zig) | O(n) core |
| Prefix Function | [`strings/prefix_function.zig`](strings/prefix_function.zig) | O(n) |
| Remove Duplicate Words | [`strings/remove_duplicate.zig`](strings/remove_duplicate.zig) | O(k log k + n) |
| Reverse Letters | [`strings/reverse_letters.zig`](strings/reverse_letters.zig) | O(n) |
| Snake Case to Camel/Pascal Case | [`strings/snake_case_to_camel_pascal_case.zig`](strings/snake_case_to_camel_pascal_case.zig) | O(n) |
| Strip | [`strings/strip.zig`](strings/strip.zig) | O(n) |
| Title Case | [`strings/title.zig`](strings/title.zig) | O(n) |
| Word Occurrence | [`strings/word_occurrence.zig`](strings/word_occurrence.zig) | O(n) |
| Pig Latin | [`strings/pig_latin.zig`](strings/pig_latin.zig) | O(n) |
| Wildcard Pattern Matching | [`strings/wildcard_pattern_matching.zig`](strings/wildcard_pattern_matching.zig) | O(n × m) |
| Top K Frequent Words | [`strings/top_k_frequent_words.zig`](strings/top_k_frequent_words.zig) | O(n + u log u) |
| Manacher | [`strings/manacher.zig`](strings/manacher.zig) | O(n) |
| Min Cost String Conversion | [`strings/min_cost_string_conversion.zig`](strings/min_cost_string_conversion.zig) | O(m × n) |

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
├── sorts/                   # 50 sorting algorithms
├── searches/                # 6 search algorithms
├── maths/                   # 81 math algorithms
├── data_structures/         # 17 data structure implementations
├── dynamic_programming/     # 17 dynamic programming algorithms
├── graphs/                  # 34 graph algorithms
├── bit_manipulation/        # 6 bit manipulation algorithms
├── conversions/             # 7 number base conversions
├── ciphers/                 # 47 cipher algorithms
├── hashing/                 # 1 hashing algorithm
├── strings/                 # 38 string algorithms
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

### 查找 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 线性查找 | [`searches/linear_search.zig`](searches/linear_search.zig) | O(n) |
| 二分查找 | [`searches/binary_search.zig`](searches/binary_search.zig) | O(log n) |
| 指数查找 | [`searches/exponential_search.zig`](searches/exponential_search.zig) | O(log n) |
| 插值查找 | [`searches/interpolation_search.zig`](searches/interpolation_search.zig) | O(log log n) 平均 |
| 跳跃查找 | [`searches/jump_search.zig`](searches/jump_search.zig) | O(√n) |
| 三分查找 | [`searches/ternary_search.zig`](searches/ternary_search.zig) | O(log₃ n) |

### 数学 (81)

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
| 平均绝对偏差 | [`maths/average_absolute_deviation.zig`](maths/average_absolute_deviation.zig) | O(n) |
| 切比雪夫距离 | [`maths/chebyshev_distance.zig`](maths/chebyshev_distance.zig) | O(n) |
| Minkowski 距离 | [`maths/minkowski_distance.zig`](maths/minkowski_distance.zig) | O(n) |
| Jaccard 相似度 | [`maths/jaccard_similarity.zig`](maths/jaccard_similarity.zig) | O(n + m) |
| 小数部分提取 | [`maths/decimal_isolate.zig`](maths/decimal_isolate.zig) | O(1) |
| 向下取整 | [`maths/floor.zig`](maths/floor.zig) | O(1) |
| 向上取整 | [`maths/ceil.zig`](maths/ceil.zig) | O(1) |
| 符号函数 | [`maths/signum.zig`](maths/signum.zig) | O(1) |
| 删一位取最大值 | [`maths/remove_digit.zig`](maths/remove_digit.zig) | O(d²) |
| 位运算加法（无算术运算符） | [`maths/addition_without_arithmetic.zig`](maths/addition_without_arithmetic.zig) | O(1) |
| 弧长计算 | [`maths/arc_length.zig`](maths/arc_length.zig) | O(1) |
| 多边形可构成性检查 | [`maths/check_polygon.zig`](maths/check_polygon.zig) | O(n log n) |
| 组合数（nCk） | [`maths/combinations.zig`](maths/combinations.zig) | O(k) |
| 双阶乘 | [`maths/double_factorial.zig`](maths/double_factorial.zig) | O(n) |
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

### 图算法 (34)

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
| 割点检测 | [`graphs/articulation_points.zig`](graphs/articulation_points.zig) | O(V + E) |
| Kosaraju 强连通分量 | [`graphs/kosaraju_scc.zig`](graphs/kosaraju_scc.zig) | O(V + E) |
| Kahn 拓扑排序 | [`graphs/kahn_topological_sort.zig`](graphs/kahn_topological_sort.zig) | O(V + E) |
| BFS 最短路径 | [`graphs/breadth_first_search_shortest_path.zig`](graphs/breadth_first_search_shortest_path.zig) | O(V + E) |
| Boruvka 最小生成树 | [`graphs/boruvka_mst.zig`](graphs/boruvka_mst.zig) | O(E log V) |
| 0-1 BFS 最短路径 | [`graphs/zero_one_bfs_shortest_path.zig`](graphs/zero_one_bfs_shortest_path.zig) | O(V + E) |
| 双向 BFS 路径搜索 | [`graphs/bidirectional_breadth_first_search.zig`](graphs/bidirectional_breadth_first_search.zig) | O(V + E) |
| 二值网格 Dijkstra | [`graphs/dijkstra_binary_grid.zig`](graphs/dijkstra_binary_grid.zig) | O((R·C)²) |
| Even Tree（偶数森林最大切边） | [`graphs/even_tree.zig`](graphs/even_tree.zig) | O(V + E) |
| Gale-Shapley 稳定匹配 | [`graphs/gale_shapley_stable_matching.zig`](graphs/gale_shapley_stable_matching.zig) | O(n²) |
| PageRank（迭代版） | [`graphs/page_rank.zig`](graphs/page_rank.zig) | O(iterations·(V + E)) |
| 双向 Dijkstra | [`graphs/bidirectional_dijkstra.zig`](graphs/bidirectional_dijkstra.zig) | O(V² + E) |
| 贪心最佳优先搜索 | [`graphs/greedy_best_first.zig`](graphs/greedy_best_first.zig) | O(V²) |
| Dinic 最大流 | [`graphs/dinic_max_flow.zig`](graphs/dinic_max_flow.zig) | O(V²·E) |
| 双向搜索 | [`graphs/bidirectional_search.zig`](graphs/bidirectional_search.zig) | O(V + E) |
| 最小路径和 | [`graphs/minimum_path_sum.zig`](graphs/minimum_path_sum.zig) | O(rows·cols) |
| 图深拷贝 | [`graphs/deep_clone_graph.zig`](graphs/deep_clone_graph.zig) | O(V + E) |
| Dijkstra（邻接矩阵） | [`graphs/dijkstra_matrix.zig`](graphs/dijkstra_matrix.zig) | O(V²) |

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

### 密码学 (47)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 凯撒密码 | [`ciphers/caesar_cipher.zig`](ciphers/caesar_cipher.zig) | O(n · m) |
| ROT13 / 凯撒移位 | [`ciphers/rot13.zig`](ciphers/rot13.zig) | O(n) |
| Atbash 密码 | [`ciphers/atbash.zig`](ciphers/atbash.zig) | O(n) |
| 维吉尼亚密码 | [`ciphers/vigenere_cipher.zig`](ciphers/vigenere_cipher.zig) | O(n) |
| 栅栏密码 | [`ciphers/rail_fence_cipher.zig`](ciphers/rail_fence_cipher.zig) | O(n) |
| XOR 密码 | [`ciphers/xor_cipher.zig`](ciphers/xor_cipher.zig) | O(n) |
| Base64 编解码 | [`ciphers/base64_cipher.zig`](ciphers/base64_cipher.zig) | O(n) |
| 列换位密码（Route） | [`ciphers/transposition_cipher.zig`](ciphers/transposition_cipher.zig) | O(n) |
| A1Z26 字母数字密码 | [`ciphers/a1z26.zig`](ciphers/a1z26.zig) | O(n) |
| 仿射密码 | [`ciphers/affine_cipher.zig`](ciphers/affine_cipher.zig) | O(n · m) |
| Baconian 密码 | [`ciphers/baconian_cipher.zig`](ciphers/baconian_cipher.zig) | O(n) |
| Base16 编解码 | [`ciphers/base16.zig`](ciphers/base16.zig) | O(n) |
| Base32 编解码 | [`ciphers/base32.zig`](ciphers/base32.zig) | O(n) |
| Base85 编解码 | [`ciphers/base85.zig`](ciphers/base85.zig) | O(n) |
| 摩尔斯密码 | [`ciphers/morse_code.zig`](ciphers/morse_code.zig) | O(n · table) |
| 波利比奥斯方阵密码 | [`ciphers/polybius.zig`](ciphers/polybius.zig) | O(n) |
| 自动密钥密码 | [`ciphers/autokey.zig`](ciphers/autokey.zig) | O(n) |
| Beaufort 密码 | [`ciphers/beaufort_cipher.zig`](ciphers/beaufort_cipher.zig) | O(n) |
| Gronsfeld 密码 | [`ciphers/gronsfeld_cipher.zig`](ciphers/gronsfeld_cipher.zig) | O(n) |
| Vernam 密码 | [`ciphers/vernam_cipher.zig`](ciphers/vernam_cipher.zig) | O(n) |
| Running Key 密码 | [`ciphers/running_key_cipher.zig`](ciphers/running_key_cipher.zig) | O(n) |
| Onepad 密码 | [`ciphers/onepad_cipher.zig`](ciphers/onepad_cipher.zig) | O(n) |
| 置换密码 | [`ciphers/permutation_cipher.zig`](ciphers/permutation_cipher.zig) | O(n) |
| 单表代换密码 | [`ciphers/mono_alphabetic_ciphers.zig`](ciphers/mono_alphabetic_ciphers.zig) | O(n) |
| 凯撒暴力解密 | [`ciphers/brute_force_caesar_cipher.zig`](ciphers/brute_force_caesar_cipher.zig) | O(26 · n) |
| Cryptomath 模逆计算 | [`ciphers/cryptomath_module.zig`](ciphers/cryptomath_module.zig) | O(log m) |
| Diffie 原根搜索 | [`ciphers/diffie.zig`](ciphers/diffie.zig) | O(m² log m) |
| 确定性 Miller-Rabin | [`ciphers/deterministic_miller_rabin.zig`](ciphers/deterministic_miller_rabin.zig) | O(k · log³ n) |
| RSA 因子分解（d,e,n） | [`ciphers/rsa_factorization.zig`](ciphers/rsa_factorization.zig) | 随机化 |
| Porta 密码 | [`ciphers/porta_cipher.zig`](ciphers/porta_cipher.zig) | O(n) |
| 混合关键词密码 | [`ciphers/mixed_keyword_cypher.zig`](ciphers/mixed_keyword_cypher.zig) | O(n) |
| 简单关键词密码 | [`ciphers/simple_keyword_cypher.zig`](ciphers/simple_keyword_cypher.zig) | O(n) |
| 简单替换密码 | [`ciphers/simple_substitution_cipher.zig`](ciphers/simple_substitution_cipher.zig) | O(n) |
| Rabin-Miller 素性测试 | [`ciphers/rabin_miller.zig`](ciphers/rabin_miller.zig) | O(k · log³ n) |
| RSA 密钥生成 | [`ciphers/rsa_key_generator.zig`](ciphers/rsa_key_generator.zig) | 概率型 |
| RSA 加解密 | [`ciphers/rsa_cipher.zig`](ciphers/rsa_cipher.zig) | O(块数 · log 指数) |
| ElGamal 密钥生成 | [`ciphers/elgamal_key_generator.zig`](ciphers/elgamal_key_generator.zig) | 概率型 |
| 置换密码文件封装 | [`ciphers/transposition_cipher_encrypt_decrypt_file.zig`](ciphers/transposition_cipher_encrypt_decrypt_file.zig) | O(n) |
| Bifid 密码 | [`ciphers/bifid.zig`](ciphers/bifid.zig) | O(n) |
| Playfair 密码 | [`ciphers/playfair_cipher.zig`](ciphers/playfair_cipher.zig) | O(n) |
| 凯撒卡方解密 | [`ciphers/decrypt_caesar_with_chi_squared.zig`](ciphers/decrypt_caesar_with_chi_squared.zig) | O(26 · n²) |
| 分式摩尔斯密码 | [`ciphers/fractionated_morse_cipher.zig`](ciphers/fractionated_morse_cipher.zig) | O(n) |
| Hill 密码 | [`ciphers/hill_cipher.zig`](ciphers/hill_cipher.zig) | O(n) |
| Shuffled Shift 密码 | [`ciphers/shuffled_shift_cipher.zig`](ciphers/shuffled_shift_cipher.zig) | O(n · m) |
| Trifid 密码 | [`ciphers/trifid_cipher.zig`](ciphers/trifid_cipher.zig) | O(n) |
| Enigma 机器 2 | [`ciphers/enigma_machine2.zig`](ciphers/enigma_machine2.zig) | O(n · 26) |
| Diffie-Hellman 密钥交换 | [`ciphers/diffie_hellman.zig`](ciphers/diffie_hellman.zig) | 每次幂运算 O(log exp) |

### 哈希 (1)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| SHA-256 | [`hashing/sha256.zig`](hashing/sha256.zig) | O(n) |

### 字符串 (38)

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
| 驼峰转下划线 | [`strings/camel_case_to_snake_case.zig`](strings/camel_case_to_snake_case.zig) | O(n) |
| 回文重排可行性检查 | [`strings/can_string_be_rearranged_as_palindrome.zig`](strings/can_string_be_rearranged_as_palindrome.zig) | O(n) |
| 首字母大写 | [`strings/capitalize.zig`](strings/capitalize.zig) | O(n) |
| 统计元音数量 | [`strings/count_vowels.zig`](strings/count_vowels.zig) | O(n) |
| 唯一字符检查 | [`strings/is_contains_unique_chars.zig`](strings/is_contains_unique_chars.zig) | O(n) |
| 同构词（Isogram）检查 | [`strings/is_isogram.zig`](strings/is_isogram.zig) | O(n) |
| 字符串连接 | [`strings/join.zig`](strings/join.zig) | O(total_len) |
| 转小写（ASCII） | [`strings/lower.zig`](strings/lower.zig) | O(n) |
| 字符串分割 | [`strings/split.zig`](strings/split.zig) | O(n) |
| 转大写（ASCII） | [`strings/upper.zig`](strings/upper.zig) | O(n) |
| 交替合并字符串 | [`strings/alternative_string_arrange.zig`](strings/alternative_string_arrange.zig) | O(n + m) |
| Boyer-Moore 搜索 | [`strings/boyer_moore_search.zig`](strings/boyer_moore_search.zig) | 最坏 O(n·m) |
| Bitap 字符串匹配 | [`strings/bitap_string_match.zig`](strings/bitap_string_match.zig) | 核心 O(n) |
| 前缀函数 | [`strings/prefix_function.zig`](strings/prefix_function.zig) | O(n) |
| 去重单词 | [`strings/remove_duplicate.zig`](strings/remove_duplicate.zig) | O(k log k + n) |
| 反转长单词字母 | [`strings/reverse_letters.zig`](strings/reverse_letters.zig) | O(n) |
| 下划线转驼峰/帕斯卡 | [`strings/snake_case_to_camel_pascal_case.zig`](strings/snake_case_to_camel_pascal_case.zig) | O(n) |
| 裁剪首尾字符 | [`strings/strip.zig`](strings/strip.zig) | O(n) |
| 标题大小写转换 | [`strings/title.zig`](strings/title.zig) | O(n) |
| 单词出现次数统计 | [`strings/word_occurrence.zig`](strings/word_occurrence.zig) | O(n) |
| 猪拉丁文转换 | [`strings/pig_latin.zig`](strings/pig_latin.zig) | O(n) |
| 通配模式匹配（. 和 *） | [`strings/wildcard_pattern_matching.zig`](strings/wildcard_pattern_matching.zig) | O(n × m) |
| Top K 高频词 | [`strings/top_k_frequent_words.zig`](strings/top_k_frequent_words.zig) | O(n + u log u) |
| Manacher 最长回文子串 | [`strings/manacher.zig`](strings/manacher.zig) | O(n) |
| 最小代价字符串转换 | [`strings/min_cost_string_conversion.zig`](strings/min_cost_string_conversion.zig) | O(m × n) |

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
├── sorts/                   # 50 种排序算法
├── searches/                # 6 种查找算法
├── maths/                   # 81 种数学算法
├── data_structures/         # 17 种数据结构实现
├── dynamic_programming/     # 17 个动态规划算法
├── graphs/                  # 34 个图算法
├── bit_manipulation/        # 6 个位运算算法
├── conversions/             # 7 个进制转换
├── ciphers/                 # 47 个密码学算法
├── hashing/                 # 1 个哈希算法
├── strings/                 # 38 个字符串算法
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
