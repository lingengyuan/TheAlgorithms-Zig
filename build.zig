const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 在循环外创建一次 test step，避免重复注册 panic
    const test_step = b.step("test", "Run all algorithm tests");

    // 每个算法文件注册为独立测试单元
    const test_files = [_][]const u8{
        // Sorts
        "sorts/bubble_sort.zig",
        "sorts/insertion_sort.zig",
        "sorts/merge_sort.zig",
        "sorts/quick_sort.zig",
        "sorts/heap_sort.zig",
        "sorts/radix_sort.zig",
        "sorts/bucket_sort.zig",
        "sorts/selection_sort.zig",
        "sorts/shell_sort.zig",
        "sorts/counting_sort.zig",
        "sorts/cocktail_shaker_sort.zig",
        "sorts/gnome_sort.zig",
        // Searches
        "searches/linear_search.zig",
        "searches/binary_search.zig",
        "searches/exponential_search.zig",
        "searches/interpolation_search.zig",
        "searches/jump_search.zig",
        "searches/ternary_search.zig",
        // Maths
        "maths/gcd.zig",
        "maths/lcm.zig",
        "maths/fibonacci.zig",
        "maths/prime_check.zig",
        "maths/sieve_of_eratosthenes.zig",
        "maths/power.zig",
        "maths/factorial.zig",
        "maths/collatz_sequence.zig",
        "maths/extended_euclidean.zig",
        "maths/modular_inverse.zig",
        "maths/eulers_totient.zig",
        "maths/chinese_remainder_theorem.zig",
        "maths/binomial_coefficient.zig",
        "maths/integer_square_root.zig",
        // Data Structures
        "data_structures/stack.zig",
        "data_structures/queue.zig",
        "data_structures/singly_linked_list.zig",
        "data_structures/doubly_linked_list.zig",
        "data_structures/binary_search_tree.zig",
        "data_structures/min_heap.zig",
        "data_structures/trie.zig",
        "data_structures/disjoint_set.zig",
        "data_structures/avl_tree.zig",
        "data_structures/max_heap.zig",
        "data_structures/priority_queue.zig",
        // Dynamic Programming
        "dynamic_programming/climbing_stairs.zig",
        "dynamic_programming/fibonacci_dp.zig",
        "dynamic_programming/coin_change.zig",
        "dynamic_programming/max_subarray_sum.zig",
        "dynamic_programming/longest_common_subsequence.zig",
        "dynamic_programming/edit_distance.zig",
        "dynamic_programming/knapsack.zig",
        "dynamic_programming/longest_increasing_subsequence.zig",
        "dynamic_programming/rod_cutting.zig",
        "dynamic_programming/matrix_chain_multiplication.zig",
        "dynamic_programming/palindrome_partitioning.zig",
        "dynamic_programming/word_break.zig",
        "dynamic_programming/catalan_numbers.zig",
        // Graphs
        "graphs/bfs.zig",
        "graphs/dfs.zig",
        "graphs/dijkstra.zig",
        "graphs/bellman_ford.zig",
        "graphs/topological_sort.zig",
        "graphs/floyd_warshall.zig",
        "graphs/detect_cycle.zig",
        "graphs/connected_components.zig",
        "graphs/kruskal.zig",
        "graphs/prim.zig",
        // Bit Manipulation
        "bit_manipulation/is_power_of_two.zig",
        "bit_manipulation/count_set_bits.zig",
        "bit_manipulation/find_unique_number.zig",
        "bit_manipulation/reverse_bits.zig",
        "bit_manipulation/missing_number.zig",
        "bit_manipulation/power_of_4.zig",
        // Conversions
        "conversions/decimal_to_binary.zig",
        "conversions/binary_to_decimal.zig",
        "conversions/decimal_to_hexadecimal.zig",
        "conversions/binary_to_hexadecimal.zig",
        // Greedy Methods
        "greedy_methods/best_time_to_buy_sell_stock.zig",
        "greedy_methods/minimum_coin_change.zig",
        "greedy_methods/minimum_waiting_time.zig",
        "greedy_methods/fractional_knapsack.zig",
        // Matrix
        "matrix/pascal_triangle.zig",
        "matrix/matrix_multiply.zig",
        "matrix/matrix_transpose.zig",
        "matrix/rotate_matrix.zig",
        "matrix/spiral_print.zig",
        // Backtracking
        "backtracking/permutations.zig",
        "backtracking/combinations.zig",
        "backtracking/subsets.zig",
        "backtracking/generate_parentheses.zig",
        "backtracking/n_queens.zig",
        "backtracking/sudoku_solver.zig",
        // Strings
        "strings/palindrome.zig",
        "strings/reverse_words.zig",
        "strings/anagram.zig",
        "strings/hamming_distance.zig",
        "strings/naive_string_search.zig",
        "strings/knuth_morris_pratt.zig",
        "strings/rabin_karp.zig",
        "strings/z_function.zig",
        "strings/levenshtein_distance.zig",
        "strings/is_pangram.zig",
    };

    for (test_files) |file| {
        // Zig 0.15+ API：addTest 使用 root_module + createModule 模式
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(file),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run = b.addRunArtifact(t);
        test_step.dependOn(&run.step);
    }
}
