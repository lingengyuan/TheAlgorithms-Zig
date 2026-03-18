//! Recursive Approach Knapsack - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/knapsack/recursive_approach_knapsack.py

const std = @import("std");
const testing = std.testing;

/// Naive recursive 0-1 knapsack.
///
/// Time complexity: O(2^n)
/// Space complexity: O(n)
pub fn knapsack(weights: []const i32, values: []const i32, number_of_items: usize, max_weight: i32, index: usize) i32 {
    if (index == number_of_items) return 0;

    const ans1 = knapsack(weights, values, number_of_items, max_weight, index + 1);
    const ans2 = if (weights[index] <= max_weight)
        values[index] + knapsack(weights, values, number_of_items, max_weight - weights[index], index + 1)
    else
        0;
    return @max(ans1, ans2);
}

test "recursive knapsack: python reference" {
    try testing.expectEqual(@as(i32, 13), knapsack(&[_]i32{ 1, 2, 4, 5 }, &[_]i32{ 5, 4, 8, 6 }, 4, 5, 0));
    try testing.expectEqual(@as(i32, 27), knapsack(&[_]i32{ 3, 4, 5 }, &[_]i32{ 10, 9, 8 }, 3, 25, 0));
}

test "recursive knapsack: boundaries" {
    try testing.expectEqual(@as(i32, 0), knapsack(&[_]i32{}, &[_]i32{}, 0, 10, 0));
    try testing.expectEqual(@as(i32, 0), knapsack(&[_]i32{5}, &[_]i32{10}, 1, 0, 0));
}
