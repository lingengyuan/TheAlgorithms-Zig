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
        // Data Structures
        "data_structures/stack.zig",
        "data_structures/queue.zig",
        "data_structures/singly_linked_list.zig",
        "data_structures/doubly_linked_list.zig",
        "data_structures/binary_search_tree.zig",
        "data_structures/min_heap.zig",
        // Dynamic Programming
        "dynamic_programming/climbing_stairs.zig",
        "dynamic_programming/fibonacci_dp.zig",
        "dynamic_programming/coin_change.zig",
        "dynamic_programming/max_subarray_sum.zig",
        "dynamic_programming/longest_common_subsequence.zig",
        "dynamic_programming/edit_distance.zig",
        "dynamic_programming/knapsack.zig",
        // Graphs
        "graphs/bfs.zig",
        "graphs/dfs.zig",
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
