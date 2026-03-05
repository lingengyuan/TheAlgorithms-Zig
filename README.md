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

### Data Structures (101)

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
| Sparse Table (RMQ) | [`data_structures/sparse_table.zig`](data_structures/sparse_table.zig) | Build O(n log n), Query O(1) |
| Bloom Filter | [`data_structures/bloom_filter.zig`](data_structures/bloom_filter.zig) | O(k) add/contains, k=2 |
| Circular Linked List | [`data_structures/circular_linked_list.zig`](data_structures/circular_linked_list.zig) | O(1) head/tail, O(n) indexed ops |
| Circular Queue (Fixed Capacity) | [`data_structures/circular_queue.zig`](data_structures/circular_queue.zig) | O(1) enqueue/dequeue |
| Queue by Two Stacks | [`data_structures/queue_by_two_stacks.zig`](data_structures/queue_by_two_stacks.zig) | Amortized O(1) put/get |
| Stack Using Two Queues | [`data_structures/stack_using_two_queues.zig`](data_structures/stack_using_two_queues.zig) | Push O(n), Pop O(1) |
| Treap | [`data_structures/treap.zig`](data_structures/treap.zig) | O(log n) avg insert/search/delete |
| Skip List | [`data_structures/skip_list.zig`](data_structures/skip_list.zig) | O(log n) avg insert/search/delete |
| Linked Queue | [`data_structures/linked_queue.zig`](data_structures/linked_queue.zig) | O(1) put/get |
| Queue by List | [`data_structures/queue_by_list.zig`](data_structures/queue_by_list.zig) | Put O(1), Get O(n) |
| Queue on Pseudo Stack | [`data_structures/queue_on_pseudo_stack.zig`](data_structures/queue_on_pseudo_stack.zig) | Put O(1), Get/Front O(n) |
| Circular Queue (Linked List) | [`data_structures/circular_queue_linked_list.zig`](data_structures/circular_queue_linked_list.zig) | O(1) enqueue/dequeue |
| Priority Queue Using List | [`data_structures/priority_queue_using_list.zig`](data_structures/priority_queue_using_list.zig) | Fixed: O(1) dequeue, Element: O(n) dequeue |
| Stack With Singly Linked List | [`data_structures/stack_with_singly_linked_list.zig`](data_structures/stack_with_singly_linked_list.zig) | O(1) push/pop |
| Stack With Doubly Linked List | [`data_structures/stack_with_doubly_linked_list.zig`](data_structures/stack_with_doubly_linked_list.zig) | O(1) push/pop |
| Deque Doubly (Linked) | [`data_structures/deque_doubly.zig`](data_structures/deque_doubly.zig) | O(1) add/remove both ends |
| Linked List From Sequence | [`data_structures/linked_list_from_sequence.zig`](data_structures/linked_list_from_sequence.zig) | O(n) |
| Middle Element Of Linked List | [`data_structures/middle_element_of_linked_list.zig`](data_structures/middle_element_of_linked_list.zig) | O(n) |
| Linked List Print Reverse | [`data_structures/linked_list_print_reverse.zig`](data_structures/linked_list_print_reverse.zig) | O(n) |
| Linked List Swap Nodes | [`data_structures/linked_list_swap_nodes.zig`](data_structures/linked_list_swap_nodes.zig) | O(n) |
| Linked List Merge Two Lists | [`data_structures/linked_list_merge_two_lists.zig`](data_structures/linked_list_merge_two_lists.zig) | O((n+m) log(n+m)) |
| Linked List Rotate To Right | [`data_structures/linked_list_rotate_to_right.zig`](data_structures/linked_list_rotate_to_right.zig) | O(n) |
| Linked List Palindrome | [`data_structures/linked_list_palindrome.zig`](data_structures/linked_list_palindrome.zig) | O(n) |
| Linked List Has Loop | [`data_structures/linked_list_has_loop.zig`](data_structures/linked_list_has_loop.zig) | O(n) |
| Balanced Parentheses | [`data_structures/balanced_parentheses.zig`](data_structures/balanced_parentheses.zig) | O(n) |
| Next Greater Element | [`data_structures/next_greater_element.zig`](data_structures/next_greater_element.zig) | O(n) |
| Largest Rectangle Histogram | [`data_structures/largest_rectangle_histogram.zig`](data_structures/largest_rectangle_histogram.zig) | O(n) |
| Stock Span Problem | [`data_structures/stock_span_problem.zig`](data_structures/stock_span_problem.zig) | O(n) |
| Postfix Evaluation | [`data_structures/postfix_evaluation.zig`](data_structures/postfix_evaluation.zig) | O(n) |
| Prefix Evaluation | [`data_structures/prefix_evaluation.zig`](data_structures/prefix_evaluation.zig) | O(n) |
| Infix To Postfix Conversion | [`data_structures/infix_to_postfix_conversion.zig`](data_structures/infix_to_postfix_conversion.zig) | O(n) |
| Infix To Prefix Conversion | [`data_structures/infix_to_prefix_conversion.zig`](data_structures/infix_to_prefix_conversion.zig) | O(n) |
| Floyd's Cycle Detection | [`data_structures/floyds_cycle_detection.zig`](data_structures/floyds_cycle_detection.zig) | O(n) |
| Reverse K Group | [`data_structures/reverse_k_group.zig`](data_structures/reverse_k_group.zig) | O(n) |
| Dijkstra's Two-Stack Algorithm | [`data_structures/dijkstras_two_stack_algorithm.zig`](data_structures/dijkstras_two_stack_algorithm.zig) | O(n) |
| Lexicographical Numbers | [`data_structures/lexicographical_numbers.zig`](data_structures/lexicographical_numbers.zig) | O(n) |
| Equilibrium Index In Array | [`data_structures/equilibrium_index_in_array.zig`](data_structures/equilibrium_index_in_array.zig) | O(n) |
| Pairs With Given Sum | [`data_structures/pairs_with_given_sum.zig`](data_structures/pairs_with_given_sum.zig) | O(n) |
| Prefix Sum | [`data_structures/prefix_sum.zig`](data_structures/prefix_sum.zig) | Build O(n), Query O(1) |
| Rotate Array | [`data_structures/rotate_array.zig`](data_structures/rotate_array.zig) | O(n) |
| Monotonic Array Check | [`data_structures/monotonic_array.zig`](data_structures/monotonic_array.zig) | O(n) |
| Kth Largest Element | [`data_structures/kth_largest_element.zig`](data_structures/kth_largest_element.zig) | Average O(n) |
| Median of Two Arrays | [`data_structures/median_two_array.zig`](data_structures/median_two_array.zig) | O((n+m) log(n+m)) |
| Index 2D Array In 1D | [`data_structures/index_2d_array_in_1d.zig`](data_structures/index_2d_array_in_1d.zig) | O(rows) |
| Find Triplets With 0 Sum | [`data_structures/find_triplets_with_0_sum.zig`](data_structures/find_triplets_with_0_sum.zig) | O(n^3) / O(n^2) hashing |
| Permutations (Array Variants) | [`data_structures/permutations.zig`](data_structures/permutations.zig) | O(n! · n) |
| Product Sum (Nested Arrays) | [`data_structures/product_sum.zig`](data_structures/product_sum.zig) | O(n) |
| Double Ended Queue (Linked Nodes) | [`data_structures/double_ended_queue.zig`](data_structures/double_ended_queue.zig) | O(1) append/pop both ends |
| Basic Binary Tree Utilities | [`data_structures/basic_binary_tree.zig`](data_structures/basic_binary_tree.zig) | O(n) traversal/metrics |
| Binary Tree Mirror (Dictionary Form) | [`data_structures/binary_tree_mirror.zig`](data_structures/binary_tree_mirror.zig) | O(n) |
| Binary Tree Node Sum | [`data_structures/binary_tree_node_sum.zig`](data_structures/binary_tree_node_sum.zig) | O(n) |
| Binary Tree Path Sum | [`data_structures/binary_tree_path_sum.zig`](data_structures/binary_tree_path_sum.zig) | O(n²) worst |
| BST Floor And Ceiling | [`data_structures/floor_and_ceiling.zig`](data_structures/floor_and_ceiling.zig) | O(h) |
| Sum Tree Check | [`data_structures/is_sum_tree.zig`](data_structures/is_sum_tree.zig) | O(n) |
| Symmetric Tree Check | [`data_structures/symmetric_tree.zig`](data_structures/symmetric_tree.zig) | O(n) |
| Diameter Of Binary Tree (Node-Centered) | [`data_structures/diameter_of_binary_tree.zig`](data_structures/diameter_of_binary_tree.zig) | O(n) |
| Binary Tree Traversals | [`data_structures/binary_tree_traversals.zig`](data_structures/binary_tree_traversals.zig) | O(n) typical; zigzag O(n²) worst |
| Different Views Of Binary Tree | [`data_structures/diff_views_of_binary_tree.zig`](data_structures/diff_views_of_binary_tree.zig) | O(n log n) |
| Merge Two Binary Trees | [`data_structures/merge_two_binary_trees.zig`](data_structures/merge_two_binary_trees.zig) | O(n) |
| Number Of Possible Binary Trees | [`data_structures/number_of_possible_binary_trees.zig`](data_structures/number_of_possible_binary_trees.zig) | O(n) |
| Serialize/Deserialize Binary Tree | [`data_structures/serialize_deserialize_binary_tree.zig`](data_structures/serialize_deserialize_binary_tree.zig) | O(n) |
| Is Sorted (Local BST Rule) | [`data_structures/is_sorted.zig`](data_structures/is_sorted.zig) | O(n) |
| Mirror Binary Tree | [`data_structures/mirror_binary_tree.zig`](data_structures/mirror_binary_tree.zig) | O(n) |
| Flatten Binary Tree To Linked List | [`data_structures/flatten_binarytree_to_linkedlist.zig`](data_structures/flatten_binarytree_to_linkedlist.zig) | O(n) |
| Distribute Coins In Binary Tree | [`data_structures/distribute_coins.zig`](data_structures/distribute_coins.zig) | O(n) |
| Maximum Sum BST In Binary Tree | [`data_structures/maximum_sum_bst.zig`](data_structures/maximum_sum_bst.zig) | O(n) |
| Inorder Tree Traversal 2022 | [`data_structures/inorder_tree_traversal_2022.zig`](data_structures/inorder_tree_traversal_2022.zig) | Insert O(h), traversal O(n) |
| Binary Search Tree (Recursive) | [`data_structures/binary_search_tree_recursive.zig`](data_structures/binary_search_tree_recursive.zig) | O(h) search/insert/remove |
| Maximum Fenwick Tree | [`data_structures/maximum_fenwick_tree.zig`](data_structures/maximum_fenwick_tree.zig) | O(log² n) update/query |
| Non-Recursive Segment Tree | [`data_structures/non_recursive_segment_tree.zig`](data_structures/non_recursive_segment_tree.zig) | O(log n) update/query |
| Lazy Segment Tree (Range Assign + Max) | [`data_structures/lazy_segment_tree.zig`](data_structures/lazy_segment_tree.zig) | O(log n) update/query |
| Segment Tree (Recursive Node Form) | [`data_structures/segment_tree_other.zig`](data_structures/segment_tree_other.zig) | O(log n) update/query |
| Lowest Common Ancestor (Binary Lifting) | [`data_structures/lowest_common_ancestor.zig`](data_structures/lowest_common_ancestor.zig) | Preprocess O(n log n), query O(log n) |
| Wavelet Tree | [`data_structures/wavelet_tree.zig`](data_structures/wavelet_tree.zig) | Build O(n log sigma), query O(log sigma) |
| Alternate Disjoint Set | [`data_structures/alternate_disjoint_set.zig`](data_structures/alternate_disjoint_set.zig) | Amortized O(alpha(n)) |
| Doubly Linked List (Double Ended Variant) | [`data_structures/doubly_linked_list_two.zig`](data_structures/doubly_linked_list_two.zig) | O(1) head/tail ops, O(n) search |
| Heap (Max Heap) | [`data_structures/heap.zig`](data_structures/heap.zig) | Build O(n), push/pop O(log n) |
| Heap (Generic Item+Score) | [`data_structures/heap_generic.zig`](data_structures/heap_generic.zig) | O(log n) insert/update/delete |
| Skew Heap | [`data_structures/skew_heap.zig`](data_structures/skew_heap.zig) | Amortized O(log n) |
| Randomized Meldable Heap | [`data_structures/randomized_heap.zig`](data_structures/randomized_heap.zig) | Expected O(log n) |
| Hash Table (Linear Probing) | [`data_structures/hash_table.zig`](data_structures/hash_table.zig) | Average O(1) insert/query |
| Hash Table (Linked-List Buckets) | [`data_structures/hash_table_with_linked_list.zig`](data_structures/hash_table_with_linked_list.zig) | Average O(1) insert/query |
| Quadratic Probing Hash Table | [`data_structures/quadratic_probing.zig`](data_structures/quadratic_probing.zig) | Average O(1) insert/query |
| Radix Tree | [`data_structures/radix_tree.zig`](data_structures/radix_tree.zig) | O(L) per operation |

### Dynamic Programming (42)

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
| Combination Sum IV (Ordered Combinations) | [`dynamic_programming/combination_sum_iv.zig`](dynamic_programming/combination_sum_iv.zig) | O(target × n) |
| Minimum Steps to One | [`dynamic_programming/min_steps_to_one.zig`](dynamic_programming/min_steps_to_one.zig) | O(n) |
| Minimum Cost Path (Grid) | [`dynamic_programming/minimum_cost_path.zig`](dynamic_programming/minimum_cost_path.zig) | O(rows × cols) |
| Minimum Tickets Cost | [`dynamic_programming/minimum_tickets_cost.zig`](dynamic_programming/minimum_tickets_cost.zig) | O(365) |
| Regex Match (`.` and `*`) | [`dynamic_programming/regex_match.zig`](dynamic_programming/regex_match.zig) | O(m × n) |
| Integer Partition Count | [`dynamic_programming/integer_partition.zig`](dynamic_programming/integer_partition.zig) | O(n²) |
| Tribonacci Sequence | [`dynamic_programming/tribonacci.zig`](dynamic_programming/tribonacci.zig) | O(n) |
| Maximum Non-Adjacent Sum | [`dynamic_programming/max_non_adjacent_sum.zig`](dynamic_programming/max_non_adjacent_sum.zig) | O(n) |
| Minimum Partition Difference | [`dynamic_programming/minimum_partition.zig`](dynamic_programming/minimum_partition.zig) | O(n × total_sum) |
| Minimum Squares To Represent A Number | [`dynamic_programming/minimum_squares_to_represent_a_number.zig`](dynamic_programming/minimum_squares_to_represent_a_number.zig) | O(n × sqrt(n)) |
| Longest Common Substring | [`dynamic_programming/longest_common_substring.zig`](dynamic_programming/longest_common_substring.zig) | O(m × n) |
| Largest Divisible Subset | [`dynamic_programming/largest_divisible_subset.zig`](dynamic_programming/largest_divisible_subset.zig) | O(n²) |
| Optimal Binary Search Tree (Cost) | [`dynamic_programming/optimal_binary_search_tree.zig`](dynamic_programming/optimal_binary_search_tree.zig) | O(n²) |
| Range Sum Query (Prefix Sum) | [`dynamic_programming/range_sum_query.zig`](dynamic_programming/range_sum_query.zig) | O(n + q) |
| Minimum Size Subarray Sum | [`dynamic_programming/minimum_size_subarray_sum.zig`](dynamic_programming/minimum_size_subarray_sum.zig) | O(n) |
| Abbreviation DP | [`dynamic_programming/abbreviation.zig`](dynamic_programming/abbreviation.zig) | O(n × m) |
| Matrix Chain Order (Cost + Split Tables) | [`dynamic_programming/matrix_chain_order.zig`](dynamic_programming/matrix_chain_order.zig) | O(n³) |
| Min Distance Up-Bottom (Top-Down Edit Distance) | [`dynamic_programming/min_distance_up_bottom.zig`](dynamic_programming/min_distance_up_bottom.zig) | O(m × n) |
| Trapped Rainwater | [`dynamic_programming/trapped_water.zig`](dynamic_programming/trapped_water.zig) | O(n) |
| Iterating Through Submasks | [`dynamic_programming/iterating_through_submasks.zig`](dynamic_programming/iterating_through_submasks.zig) | O(2^k) |
| Fast Fibonacci (Doubling) | [`dynamic_programming/fast_fibonacci.zig`](dynamic_programming/fast_fibonacci.zig) | O(log n) |
| Fizz Buzz | [`dynamic_programming/fizz_buzz.zig`](dynamic_programming/fizz_buzz.zig) | O(iterations) |
| LIS Iterative (Sequence, O(n²)) | [`dynamic_programming/longest_increasing_subsequence_iterative.zig`](dynamic_programming/longest_increasing_subsequence_iterative.zig) | O(n²) |
| LIS Length (O(n log n)) | [`dynamic_programming/longest_increasing_subsequence_o_nlogn.zig`](dynamic_programming/longest_increasing_subsequence_o_nlogn.zig) | O(n log n) |
| Assignment Using Bitmask | [`dynamic_programming/bitmask.zig`](dynamic_programming/bitmask.zig) | O(2^P · T · avg_deg) |

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

### Backtracking (17)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Permutations | [`backtracking/permutations.zig`](backtracking/permutations.zig) | O(n! · n) |
| Combinations | [`backtracking/combinations.zig`](backtracking/combinations.zig) | O(C(n,k)) |
| Subsets | [`backtracking/subsets.zig`](backtracking/subsets.zig) | O(2ⁿ) |
| Generate Parentheses | [`backtracking/generate_parentheses.zig`](backtracking/generate_parentheses.zig) | O(Catalan(n)) |
| N-Queens | [`backtracking/n_queens.zig`](backtracking/n_queens.zig) | O(n!) |
| Sudoku Solver | [`backtracking/sudoku_solver.zig`](backtracking/sudoku_solver.zig) | O(9^m) |
| Word Search | [`backtracking/word_search.zig`](backtracking/word_search.zig) | O(rows · cols · 4^L) |
| Rat in a Maze | [`backtracking/rat_in_maze.zig`](backtracking/rat_in_maze.zig) | worst-case exponential |
| Combination Sum | [`backtracking/combination_sum.zig`](backtracking/combination_sum.zig) | worst-case exponential |
| Power Sum | [`backtracking/power_sum.zig`](backtracking/power_sum.zig) | worst-case exponential |
| Word Break (Backtracking) | [`backtracking/word_break.zig`](backtracking/word_break.zig) | worst-case exponential |
| Sum of Subsets | [`backtracking/sum_of_subsets.zig`](backtracking/sum_of_subsets.zig) | worst-case exponential |
| Hamiltonian Cycle | [`backtracking/hamiltonian_cycle.zig`](backtracking/hamiltonian_cycle.zig) | worst-case exponential |
| All Subsequences | [`backtracking/all_subsequences.zig`](backtracking/all_subsequences.zig) | O(2ⁿ) |
| Match Word Pattern | [`backtracking/match_word_pattern.zig`](backtracking/match_word_pattern.zig) | worst-case exponential |
| Minimax | [`backtracking/minimax.zig`](backtracking/minimax.zig) | O(2^h) |
| Graph Coloring (M-Coloring) | [`backtracking/coloring.zig`](backtracking/coloring.zig) | O(m^n) |

### Bit Manipulation (17)

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

### Conversions (18)

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

### Boolean Algebra (11)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| AND Gate | [`boolean_algebra/and_gate.zig`](boolean_algebra/and_gate.zig) | O(1) / O(n) |
| OR Gate | [`boolean_algebra/or_gate.zig`](boolean_algebra/or_gate.zig) | O(1) |
| XOR Gate | [`boolean_algebra/xor_gate.zig`](boolean_algebra/xor_gate.zig) | O(1) |
| NAND Gate | [`boolean_algebra/nand_gate.zig`](boolean_algebra/nand_gate.zig) | O(1) |
| NOR Gate | [`boolean_algebra/nor_gate.zig`](boolean_algebra/nor_gate.zig) | O(1) |
| NOT Gate | [`boolean_algebra/not_gate.zig`](boolean_algebra/not_gate.zig) | O(1) |
| XNOR Gate | [`boolean_algebra/xnor_gate.zig`](boolean_algebra/xnor_gate.zig) | O(1) |
| IMPLY Gate | [`boolean_algebra/imply_gate.zig`](boolean_algebra/imply_gate.zig) | O(1) / O(n) |
| NIMPLY Gate | [`boolean_algebra/nimply_gate.zig`](boolean_algebra/nimply_gate.zig) | O(1) |
| 2-to-1 Multiplexer | [`boolean_algebra/multiplexer.zig`](boolean_algebra/multiplexer.zig) | O(1) |
| Karnaugh Map Simplification | [`boolean_algebra/karnaugh_map_simplification.zig`](boolean_algebra/karnaugh_map_simplification.zig) | O(r · c) |

### Divide and Conquer (11)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Maximum Subarray (Divide and Conquer) | [`divide_and_conquer/max_subarray.zig`](divide_and_conquer/max_subarray.zig) | O(n log n) |
| Peak of Unimodal Array | [`divide_and_conquer/peak.zig`](divide_and_conquer/peak.zig) | O(log n) |
| Fast Power | [`divide_and_conquer/power.zig`](divide_and_conquer/power.zig) | O(log |b|) |
| Kth Order Statistic | [`divide_and_conquer/kth_order_statistic.zig`](divide_and_conquer/kth_order_statistic.zig) | O(n) average |
| Inversion Count | [`divide_and_conquer/inversions.zig`](divide_and_conquer/inversions.zig) | O(n log n) |
| Max Difference Pair | [`divide_and_conquer/max_difference_pair.zig`](divide_and_conquer/max_difference_pair.zig) | O(n log n) |
| Merge Sort (Divide and Conquer) | [`divide_and_conquer/mergesort.zig`](divide_and_conquer/mergesort.zig) | O(n log n) |
| Heap's Algorithm (Permutations) | [`divide_and_conquer/heaps_algorithm.zig`](divide_and_conquer/heaps_algorithm.zig) | O(n · n!) |
| Heap's Algorithm (Iterative) | [`divide_and_conquer/heaps_algorithm_iterative.zig`](divide_and_conquer/heaps_algorithm_iterative.zig) | O(n · n!) |
| Closest Pair of Points | [`divide_and_conquer/closest_pair_of_points.zig`](divide_and_conquer/closest_pair_of_points.zig) | O(n log n) |
| Convex Hull | [`divide_and_conquer/convex_hull.zig`](divide_and_conquer/convex_hull.zig) | O(n log n) |

### Linear Algebra (11)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Gaussian Elimination | [`linear_algebra/gaussian_elimination.zig`](linear_algebra/gaussian_elimination.zig) | O(n³) |
| LU Decomposition | [`linear_algebra/lu_decomposition.zig`](linear_algebra/lu_decomposition.zig) | O(n³) |
| Jacobi Iteration Method | [`linear_algebra/jacobi_iteration_method.zig`](linear_algebra/jacobi_iteration_method.zig) | O(iterations · n²) |
| Matrix Inversion | [`linear_algebra/matrix_inversion.zig`](linear_algebra/matrix_inversion.zig) | O(n³) |
| Rank of Matrix | [`linear_algebra/rank_of_matrix.zig`](linear_algebra/rank_of_matrix.zig) | O(min(r,c)·r·c) |
| Rayleigh Quotient | [`linear_algebra/rayleigh_quotient.zig`](linear_algebra/rayleigh_quotient.zig) | O(n²) |
| Power Iteration | [`linear_algebra/power_iteration.zig`](linear_algebra/power_iteration.zig) | O(iterations · n²) |
| Schur Complement | [`linear_algebra/schur_complement.zig`](linear_algebra/schur_complement.zig) | O(n³ + n²m + nm²) |
| 2D Transformations | [`linear_algebra/transformations_2d.zig`](linear_algebra/transformations_2d.zig) | O(1) |
| Gaussian Elimination (Pivoting) | [`linear_algebra/gaussian_elimination_pivoting.zig`](linear_algebra/gaussian_elimination_pivoting.zig) | O(n³) |
| Conjugate Gradient Method | [`linear_algebra/conjugate_gradient.zig`](linear_algebra/conjugate_gradient.zig) | O(iterations · n²) |

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

Note:
- `ciphers/diffie_hellman.zig` currently uses toy safe-prime groups (keeping group-id API shape) instead of RFC3526 huge primes, because this repository phase focuses on algorithm behavior validation under Zig `u128` without adding a big-integer dependency.

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
├── data_structures/         # 101 data structure implementations
├── dynamic_programming/     # 42 dynamic programming algorithms
├── graphs/                  # 46 graph algorithms
├── bit_manipulation/        # 17 bit manipulation algorithms
├── conversions/             # 18 conversion algorithms
├── boolean_algebra/         # 11 boolean algebra algorithms
├── divide_and_conquer/      # 11 divide-and-conquer algorithms
├── linear_algebra/          # 11 linear algebra algorithms
├── ciphers/                 # 47 cipher algorithms
├── hashing/                 # 1 hashing algorithm
├── strings/                 # 38 string algorithms
├── greedy_methods/          # 7 greedy algorithms
├── matrix/                  # 5 matrix algorithms
└── backtracking/            # 17 backtracking algorithms
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

### 数据结构 (101)

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
| 稀疏表（RMQ） | [`data_structures/sparse_table.zig`](data_structures/sparse_table.zig) | 构建 O(n log n)，查询 O(1) |
| 布隆过滤器 | [`data_structures/bloom_filter.zig`](data_structures/bloom_filter.zig) | add/contains 为 O(k)，k=2 |
| 循环链表 | [`data_structures/circular_linked_list.zig`](data_structures/circular_linked_list.zig) | 头尾操作 O(1)，按索引 O(n) |
| 循环队列（定长） | [`data_structures/circular_queue.zig`](data_structures/circular_queue.zig) | enqueue/dequeue O(1) |
| 双栈实现队列 | [`data_structures/queue_by_two_stacks.zig`](data_structures/queue_by_two_stacks.zig) | put/get 均摊 O(1) |
| 双队列实现栈 | [`data_structures/stack_using_two_queues.zig`](data_structures/stack_using_two_queues.zig) | push O(n)，pop O(1) |
| Treap（树堆） | [`data_structures/treap.zig`](data_structures/treap.zig) | 插入/查找/删除平均 O(log n) |
| 跳表 | [`data_structures/skip_list.zig`](data_structures/skip_list.zig) | 插入/查找/删除平均 O(log n) |
| 链式队列 | [`data_structures/linked_queue.zig`](data_structures/linked_queue.zig) | put/get O(1) |
| 列表实现队列 | [`data_structures/queue_by_list.zig`](data_structures/queue_by_list.zig) | put O(1)，get O(n) |
| 伪栈实现队列 | [`data_structures/queue_on_pseudo_stack.zig`](data_structures/queue_on_pseudo_stack.zig) | put O(1)，get/front O(n) |
| 循环队列（链表） | [`data_structures/circular_queue_linked_list.zig`](data_structures/circular_queue_linked_list.zig) | enqueue/dequeue O(1) |
| 列表实现优先队列 | [`data_structures/priority_queue_using_list.zig`](data_structures/priority_queue_using_list.zig) | 固定优先级出队 O(1)，元素优先级出队 O(n) |
| 单链表实现栈 | [`data_structures/stack_with_singly_linked_list.zig`](data_structures/stack_with_singly_linked_list.zig) | push/pop O(1) |
| 双链表实现栈 | [`data_structures/stack_with_doubly_linked_list.zig`](data_structures/stack_with_doubly_linked_list.zig) | push/pop O(1) |
| 双向链表双端队列 | [`data_structures/deque_doubly.zig`](data_structures/deque_doubly.zig) | 两端 add/remove O(1) |
| 序列构建链表 | [`data_structures/linked_list_from_sequence.zig`](data_structures/linked_list_from_sequence.zig) | O(n) |
| 链表中间元素 | [`data_structures/middle_element_of_linked_list.zig`](data_structures/middle_element_of_linked_list.zig) | O(n) |
| 链表逆序输出 | [`data_structures/linked_list_print_reverse.zig`](data_structures/linked_list_print_reverse.zig) | O(n) |
| 链表节点交换 | [`data_structures/linked_list_swap_nodes.zig`](data_structures/linked_list_swap_nodes.zig) | O(n) |
| 合并两个有序链表 | [`data_structures/linked_list_merge_two_lists.zig`](data_structures/linked_list_merge_two_lists.zig) | O((n+m) log(n+m)) |
| 链表右旋 | [`data_structures/linked_list_rotate_to_right.zig`](data_structures/linked_list_rotate_to_right.zig) | O(n) |
| 链表回文判断 | [`data_structures/linked_list_palindrome.zig`](data_structures/linked_list_palindrome.zig) | O(n) |
| 链表环检测 | [`data_structures/linked_list_has_loop.zig`](data_structures/linked_list_has_loop.zig) | O(n) |
| 括号平衡检查 | [`data_structures/balanced_parentheses.zig`](data_structures/balanced_parentheses.zig) | O(n) |
| 下一个更大元素 | [`data_structures/next_greater_element.zig`](data_structures/next_greater_element.zig) | O(n) |
| 柱状图最大矩形 | [`data_structures/largest_rectangle_histogram.zig`](data_structures/largest_rectangle_histogram.zig) | O(n) |
| 股票跨度问题 | [`data_structures/stock_span_problem.zig`](data_structures/stock_span_problem.zig) | O(n) |
| 后缀表达式求值 | [`data_structures/postfix_evaluation.zig`](data_structures/postfix_evaluation.zig) | O(n) |
| 前缀表达式求值 | [`data_structures/prefix_evaluation.zig`](data_structures/prefix_evaluation.zig) | O(n) |
| 中缀转后缀 | [`data_structures/infix_to_postfix_conversion.zig`](data_structures/infix_to_postfix_conversion.zig) | O(n) |
| 中缀转前缀 | [`data_structures/infix_to_prefix_conversion.zig`](data_structures/infix_to_prefix_conversion.zig) | O(n) |
| Floyd 判圈算法 | [`data_structures/floyds_cycle_detection.zig`](data_structures/floyds_cycle_detection.zig) | O(n) |
| K 组链表反转 | [`data_structures/reverse_k_group.zig`](data_structures/reverse_k_group.zig) | O(n) |
| Dijkstra 双栈求值 | [`data_structures/dijkstras_two_stack_algorithm.zig`](data_structures/dijkstras_two_stack_algorithm.zig) | O(n) |
| 字典序数字生成 | [`data_structures/lexicographical_numbers.zig`](data_structures/lexicographical_numbers.zig) | O(n) |
| 数组平衡下标 | [`data_structures/equilibrium_index_in_array.zig`](data_structures/equilibrium_index_in_array.zig) | O(n) |
| 指定和配对计数 | [`data_structures/pairs_with_given_sum.zig`](data_structures/pairs_with_given_sum.zig) | O(n) |
| 前缀和 | [`data_structures/prefix_sum.zig`](data_structures/prefix_sum.zig) | 构建 O(n)，查询 O(1) |
| 数组旋转 | [`data_structures/rotate_array.zig`](data_structures/rotate_array.zig) | O(n) |
| 单调数组检查 | [`data_structures/monotonic_array.zig`](data_structures/monotonic_array.zig) | O(n) |
| 第 k 大元素 | [`data_structures/kth_largest_element.zig`](data_structures/kth_largest_element.zig) | 平均 O(n) |
| 两数组中位数 | [`data_structures/median_two_array.zig`](data_structures/median_two_array.zig) | O((n+m) log(n+m)) |
| 二维数组一维索引 | [`data_structures/index_2d_array_in_1d.zig`](data_structures/index_2d_array_in_1d.zig) | O(rows) |
| 零和三元组 | [`data_structures/find_triplets_with_0_sum.zig`](data_structures/find_triplets_with_0_sum.zig) | O(n^3) / 哈希 O(n^2) |
| 全排列（数组版本） | [`data_structures/permutations.zig`](data_structures/permutations.zig) | O(n! · n) |
| 嵌套数组乘积和 | [`data_structures/product_sum.zig`](data_structures/product_sum.zig) | O(n) |
| 双端队列（双向链表节点） | [`data_structures/double_ended_queue.zig`](data_structures/double_ended_queue.zig) | 两端 append/pop O(1) |
| 基础二叉树工具集 | [`data_structures/basic_binary_tree.zig`](data_structures/basic_binary_tree.zig) | 遍历/度量 O(n) |
| 二叉树镜像（字典表示） | [`data_structures/binary_tree_mirror.zig`](data_structures/binary_tree_mirror.zig) | O(n) |
| 二叉树节点求和 | [`data_structures/binary_tree_node_sum.zig`](data_structures/binary_tree_node_sum.zig) | O(n) |
| 二叉树路径和计数 | [`data_structures/binary_tree_path_sum.zig`](data_structures/binary_tree_path_sum.zig) | 最坏 O(n²) |
| BST Floor / Ceiling | [`data_structures/floor_and_ceiling.zig`](data_structures/floor_and_ceiling.zig) | O(h) |
| Sum Tree 判定 | [`data_structures/is_sum_tree.zig`](data_structures/is_sum_tree.zig) | O(n) |
| 对称二叉树判定 | [`data_structures/symmetric_tree.zig`](data_structures/symmetric_tree.zig) | O(n) |
| 二叉树直径（节点中心定义） | [`data_structures/diameter_of_binary_tree.zig`](data_structures/diameter_of_binary_tree.zig) | O(n) |
| 二叉树遍历集合 | [`data_structures/binary_tree_traversals.zig`](data_structures/binary_tree_traversals.zig) | 常规 O(n)，zigzag 最坏 O(n²) |
| 二叉树多视图（左/右/上/下） | [`data_structures/diff_views_of_binary_tree.zig`](data_structures/diff_views_of_binary_tree.zig) | O(n log n) |
| 合并两棵二叉树 | [`data_structures/merge_two_binary_trees.zig`](data_structures/merge_two_binary_trees.zig) | O(n) |
| 二叉树数量（Catalan/总数） | [`data_structures/number_of_possible_binary_trees.zig`](data_structures/number_of_possible_binary_trees.zig) | O(n) |
| 二叉树序列化/反序列化 | [`data_structures/serialize_deserialize_binary_tree.zig`](data_structures/serialize_deserialize_binary_tree.zig) | O(n) |
| 有序性检查（局部 BST 规则） | [`data_structures/is_sorted.zig`](data_structures/is_sorted.zig) | O(n) |
| 二叉树镜像 | [`data_structures/mirror_binary_tree.zig`](data_structures/mirror_binary_tree.zig) | O(n) |
| 二叉树拍平为链表 | [`data_structures/flatten_binarytree_to_linkedlist.zig`](data_structures/flatten_binarytree_to_linkedlist.zig) | O(n) |
| 二叉树硬币分配 | [`data_structures/distribute_coins.zig`](data_structures/distribute_coins.zig) | O(n) |
| 二叉树中 BST 子树最大和 | [`data_structures/maximum_sum_bst.zig`](data_structures/maximum_sum_bst.zig) | O(n) |
| 中序遍历（2022 版本） | [`data_structures/inorder_tree_traversal_2022.zig`](data_structures/inorder_tree_traversal_2022.zig) | 插入 O(h)，遍历 O(n) |
| 二叉搜索树（递归实现） | [`data_structures/binary_search_tree_recursive.zig`](data_structures/binary_search_tree_recursive.zig) | 搜索/插入/删除 O(h) |
| 最大值 Fenwick 树 | [`data_structures/maximum_fenwick_tree.zig`](data_structures/maximum_fenwick_tree.zig) | 更新/查询 O(log² n) |
| 非递归线段树 | [`data_structures/non_recursive_segment_tree.zig`](data_structures/non_recursive_segment_tree.zig) | 更新/查询 O(log n) |
| 懒标记线段树（区间赋值+最大值） | [`data_structures/lazy_segment_tree.zig`](data_structures/lazy_segment_tree.zig) | 更新/查询 O(log n) |
| 线段树（递归节点实现） | [`data_structures/segment_tree_other.zig`](data_structures/segment_tree_other.zig) | 更新/查询 O(log n) |
| 最近公共祖先（倍增法） | [`data_structures/lowest_common_ancestor.zig`](data_structures/lowest_common_ancestor.zig) | 预处理 O(n log n)，查询 O(log n) |
| Wavelet 树 | [`data_structures/wavelet_tree.zig`](data_structures/wavelet_tree.zig) | 构建 O(n log sigma)，查询 O(log sigma) |
| 并查集（替代实现） | [`data_structures/alternate_disjoint_set.zig`](data_structures/alternate_disjoint_set.zig) | 均摊 O(alpha(n)) |
| 双向链表（双端版本） | [`data_structures/doubly_linked_list_two.zig`](data_structures/doubly_linked_list_two.zig) | 头尾 O(1)，查找 O(n) |
| 堆（最大堆） | [`data_structures/heap.zig`](data_structures/heap.zig) | 构建 O(n)，插入/弹出 O(log n) |
| 堆（通用 item+score） | [`data_structures/heap_generic.zig`](data_structures/heap_generic.zig) | 插入/更新/删除 O(log n) |
| 斜堆 | [`data_structures/skew_heap.zig`](data_structures/skew_heap.zig) | 均摊 O(log n) |
| 随机可并堆 | [`data_structures/randomized_heap.zig`](data_structures/randomized_heap.zig) | 期望 O(log n) |
| 哈希表（线性探测） | [`data_structures/hash_table.zig`](data_structures/hash_table.zig) | 平均 O(1) 插入/查询 |
| 哈希表（链表桶） | [`data_structures/hash_table_with_linked_list.zig`](data_structures/hash_table_with_linked_list.zig) | 平均 O(1) 插入/查询 |
| 哈希表（二次探测） | [`data_structures/quadratic_probing.zig`](data_structures/quadratic_probing.zig) | 平均 O(1) 插入/查询 |
| Radix 树（压缩前缀树） | [`data_structures/radix_tree.zig`](data_structures/radix_tree.zig) | 每次操作 O(L) |

### 动态规划 (42)

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
| 组合总和 IV（有序方案数） | [`dynamic_programming/combination_sum_iv.zig`](dynamic_programming/combination_sum_iv.zig) | O(target × n) |
| 到 1 的最少步数 | [`dynamic_programming/min_steps_to_one.zig`](dynamic_programming/min_steps_to_one.zig) | O(n) |
| 最小路径代价（网格） | [`dynamic_programming/minimum_cost_path.zig`](dynamic_programming/minimum_cost_path.zig) | O(rows × cols) |
| 最低票价 | [`dynamic_programming/minimum_tickets_cost.zig`](dynamic_programming/minimum_tickets_cost.zig) | O(365) |
| 正则匹配（`.` 与 `*`） | [`dynamic_programming/regex_match.zig`](dynamic_programming/regex_match.zig) | O(m × n) |
| 整数拆分计数 | [`dynamic_programming/integer_partition.zig`](dynamic_programming/integer_partition.zig) | O(n²) |
| Tribonacci 数列 | [`dynamic_programming/tribonacci.zig`](dynamic_programming/tribonacci.zig) | O(n) |
| 最大非相邻子序列和 | [`dynamic_programming/max_non_adjacent_sum.zig`](dynamic_programming/max_non_adjacent_sum.zig) | O(n) |
| 最小划分差值 | [`dynamic_programming/minimum_partition.zig`](dynamic_programming/minimum_partition.zig) | O(n × total_sum) |
| 表示为平方数和的最少项数 | [`dynamic_programming/minimum_squares_to_represent_a_number.zig`](dynamic_programming/minimum_squares_to_represent_a_number.zig) | O(n × sqrt(n)) |
| 最长公共子串 | [`dynamic_programming/longest_common_substring.zig`](dynamic_programming/longest_common_substring.zig) | O(m × n) |
| 最大可整除子集 | [`dynamic_programming/largest_divisible_subset.zig`](dynamic_programming/largest_divisible_subset.zig) | O(n²) |
| 最优二叉搜索树（代价） | [`dynamic_programming/optimal_binary_search_tree.zig`](dynamic_programming/optimal_binary_search_tree.zig) | O(n²) |
| 区间和查询（前缀和） | [`dynamic_programming/range_sum_query.zig`](dynamic_programming/range_sum_query.zig) | O(n + q) |
| 最短满足和子数组长度 | [`dynamic_programming/minimum_size_subarray_sum.zig`](dynamic_programming/minimum_size_subarray_sum.zig) | O(n) |
| 字符串缩写匹配 DP | [`dynamic_programming/abbreviation.zig`](dynamic_programming/abbreviation.zig) | O(n × m) |
| 矩阵链次序（代价+切分表） | [`dynamic_programming/matrix_chain_order.zig`](dynamic_programming/matrix_chain_order.zig) | O(n³) |
| 编辑距离上自顶向下版 | [`dynamic_programming/min_distance_up_bottom.zig`](dynamic_programming/min_distance_up_bottom.zig) | O(m × n) |
| 接雨水 | [`dynamic_programming/trapped_water.zig`](dynamic_programming/trapped_water.zig) | O(n) |
| 子掩码遍历 | [`dynamic_programming/iterating_through_submasks.zig`](dynamic_programming/iterating_through_submasks.zig) | O(2^k) |
| 快速斐波那契（倍增法） | [`dynamic_programming/fast_fibonacci.zig`](dynamic_programming/fast_fibonacci.zig) | O(log n) |
| Fizz Buzz | [`dynamic_programming/fizz_buzz.zig`](dynamic_programming/fizz_buzz.zig) | O(iterations) |
| 最长递增子序列迭代版（序列，O(n²)） | [`dynamic_programming/longest_increasing_subsequence_iterative.zig`](dynamic_programming/longest_increasing_subsequence_iterative.zig) | O(n²) |
| 最长递增子序列长度（O(n log n)） | [`dynamic_programming/longest_increasing_subsequence_o_nlogn.zig`](dynamic_programming/longest_increasing_subsequence_o_nlogn.zig) | O(n log n) |
| Bitmask 任务分配计数 | [`dynamic_programming/bitmask.zig`](dynamic_programming/bitmask.zig) | O(2^P · T · avg_deg) |

### 图算法 (46)

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

### 回溯算法 (17)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 全排列 | [`backtracking/permutations.zig`](backtracking/permutations.zig) | O(n! · n) |
| 组合 | [`backtracking/combinations.zig`](backtracking/combinations.zig) | O(C(n,k)) |
| 子集（幂集） | [`backtracking/subsets.zig`](backtracking/subsets.zig) | O(2ⁿ) |
| 生成括号 | [`backtracking/generate_parentheses.zig`](backtracking/generate_parentheses.zig) | O(Catalan(n)) |
| N 皇后 | [`backtracking/n_queens.zig`](backtracking/n_queens.zig) | O(n!) |
| 数独求解 | [`backtracking/sudoku_solver.zig`](backtracking/sudoku_solver.zig) | O(9^m) |
| 单词搜索 | [`backtracking/word_search.zig`](backtracking/word_search.zig) | O(rows · cols · 4^L) |
| 迷宫老鼠问题 | [`backtracking/rat_in_maze.zig`](backtracking/rat_in_maze.zig) | 最坏指数级 |
| 组合总和 | [`backtracking/combination_sum.zig`](backtracking/combination_sum.zig) | 最坏指数级 |
| 幂和问题 | [`backtracking/power_sum.zig`](backtracking/power_sum.zig) | 最坏指数级 |
| 单词拆分（回溯） | [`backtracking/word_break.zig`](backtracking/word_break.zig) | 最坏指数级 |
| 子集和问题 | [`backtracking/sum_of_subsets.zig`](backtracking/sum_of_subsets.zig) | 最坏指数级 |
| 哈密顿回路 | [`backtracking/hamiltonian_cycle.zig`](backtracking/hamiltonian_cycle.zig) | 最坏指数级 |
| 全部子序列 | [`backtracking/all_subsequences.zig`](backtracking/all_subsequences.zig) | O(2ⁿ) |
| 单词模式匹配（回溯） | [`backtracking/match_word_pattern.zig`](backtracking/match_word_pattern.zig) | 最坏指数级 |
| 极小化极大算法 | [`backtracking/minimax.zig`](backtracking/minimax.zig) | O(2^h) |
| 图着色（M 着色） | [`backtracking/coloring.zig`](backtracking/coloring.zig) | O(m^n) |

### 位运算 (17)

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

### 进制转换 (18)

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

### 布尔代数 (11)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| AND 门 | [`boolean_algebra/and_gate.zig`](boolean_algebra/and_gate.zig) | O(1) / O(n) |
| OR 门 | [`boolean_algebra/or_gate.zig`](boolean_algebra/or_gate.zig) | O(1) |
| XOR 门 | [`boolean_algebra/xor_gate.zig`](boolean_algebra/xor_gate.zig) | O(1) |
| NAND 门 | [`boolean_algebra/nand_gate.zig`](boolean_algebra/nand_gate.zig) | O(1) |
| NOR 门 | [`boolean_algebra/nor_gate.zig`](boolean_algebra/nor_gate.zig) | O(1) |
| NOT 门 | [`boolean_algebra/not_gate.zig`](boolean_algebra/not_gate.zig) | O(1) |
| XNOR 门 | [`boolean_algebra/xnor_gate.zig`](boolean_algebra/xnor_gate.zig) | O(1) |
| IMPLY 门 | [`boolean_algebra/imply_gate.zig`](boolean_algebra/imply_gate.zig) | O(1) / O(n) |
| NIMPLY 门 | [`boolean_algebra/nimply_gate.zig`](boolean_algebra/nimply_gate.zig) | O(1) |
| 2选1 多路复用器 | [`boolean_algebra/multiplexer.zig`](boolean_algebra/multiplexer.zig) | O(1) |
| 卡诺图化简 | [`boolean_algebra/karnaugh_map_simplification.zig`](boolean_algebra/karnaugh_map_simplification.zig) | O(r · c) |

### 分治 (11)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 最大子数组（分治） | [`divide_and_conquer/max_subarray.zig`](divide_and_conquer/max_subarray.zig) | O(n log n) |
| 单峰数组峰值 | [`divide_and_conquer/peak.zig`](divide_and_conquer/peak.zig) | O(log n) |
| 快速幂（分治） | [`divide_and_conquer/power.zig`](divide_and_conquer/power.zig) | O(log |b|) |
| 第 k 小元素（分治） | [`divide_and_conquer/kth_order_statistic.zig`](divide_and_conquer/kth_order_statistic.zig) | 平均 O(n) |
| 逆序对计数 | [`divide_and_conquer/inversions.zig`](divide_and_conquer/inversions.zig) | O(n log n) |
| 最大差值对（分治） | [`divide_and_conquer/max_difference_pair.zig`](divide_and_conquer/max_difference_pair.zig) | O(n log n) |
| 归并排序（分治） | [`divide_and_conquer/mergesort.zig`](divide_and_conquer/mergesort.zig) | O(n log n) |
| Heap 排列算法（分治） | [`divide_and_conquer/heaps_algorithm.zig`](divide_and_conquer/heaps_algorithm.zig) | O(n · n!) |
| Heap 排列算法（迭代） | [`divide_and_conquer/heaps_algorithm_iterative.zig`](divide_and_conquer/heaps_algorithm_iterative.zig) | O(n · n!) |
| 最近点对（分治） | [`divide_and_conquer/closest_pair_of_points.zig`](divide_and_conquer/closest_pair_of_points.zig) | O(n log n) |
| 凸包（分治） | [`divide_and_conquer/convex_hull.zig`](divide_and_conquer/convex_hull.zig) | O(n log n) |

### 线性代数 (11)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 高斯消元 | [`linear_algebra/gaussian_elimination.zig`](linear_algebra/gaussian_elimination.zig) | O(n³) |
| LU 分解 | [`linear_algebra/lu_decomposition.zig`](linear_algebra/lu_decomposition.zig) | O(n³) |
| Jacobi 迭代法 | [`linear_algebra/jacobi_iteration_method.zig`](linear_algebra/jacobi_iteration_method.zig) | O(iterations · n²) |
| 矩阵求逆 | [`linear_algebra/matrix_inversion.zig`](linear_algebra/matrix_inversion.zig) | O(n³) |
| 矩阵秩 | [`linear_algebra/rank_of_matrix.zig`](linear_algebra/rank_of_matrix.zig) | O(min(r,c)·r·c) |
| 瑞利商 | [`linear_algebra/rayleigh_quotient.zig`](linear_algebra/rayleigh_quotient.zig) | O(n²) |
| 幂迭代法 | [`linear_algebra/power_iteration.zig`](linear_algebra/power_iteration.zig) | O(iterations · n²) |
| Schur 补 | [`linear_algebra/schur_complement.zig`](linear_algebra/schur_complement.zig) | O(n³ + n²m + nm²) |
| 二维变换矩阵 | [`linear_algebra/transformations_2d.zig`](linear_algebra/transformations_2d.zig) | O(1) |
| 高斯消元（部分主元） | [`linear_algebra/gaussian_elimination_pivoting.zig`](linear_algebra/gaussian_elimination_pivoting.zig) | O(n³) |
| 共轭梯度法 | [`linear_algebra/conjugate_gradient.zig`](linear_algebra/conjugate_gradient.zig) | O(iterations · n²) |

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

说明：
- `ciphers/diffie_hellman.zig` 当前采用 toy-safe 质数组（保留 group id 入口形态），未直接落 RFC3526 超大素数，原因是本阶段优先在不引入大整数依赖的前提下完成算法行为验证。

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
├── data_structures/         # 101 种数据结构实现
├── dynamic_programming/     # 42 个动态规划算法
├── graphs/                  # 46 个图算法
├── bit_manipulation/        # 17 个位运算算法
├── conversions/             # 18 个进制转换算法
├── boolean_algebra/         # 11 个布尔代数算法
├── divide_and_conquer/      # 11 个分治算法
├── linear_algebra/          # 11 个线性代数算法
├── ciphers/                 # 47 个密码学算法
├── hashing/                 # 1 个哈希算法
├── strings/                 # 38 个字符串算法
├── greedy_methods/          # 7 个贪心算法
├── matrix/                  # 5 个矩阵算法
└── backtracking/            # 17 个回溯算法
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
