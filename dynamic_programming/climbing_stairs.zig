//! Climbing Stairs - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/climbing_stairs.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of distinct ways to climb `n` steps
/// when each move can be either 1 or 2 steps.
/// Time complexity: O(n), Space complexity: O(1)
pub fn climbingStairs(n: u32) u64 {
    if (n == 0) return 1;
    if (n == 1) return 1;

    var prev2: u64 = 1;
    var prev1: u64 = 1;

    var i: u32 = 2;
    while (i <= n) : (i += 1) {
        const current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    return prev1;
}

test "climbing stairs: base cases" {
    try testing.expectEqual(@as(u64, 1), climbingStairs(0));
    try testing.expectEqual(@as(u64, 1), climbingStairs(1));
    try testing.expectEqual(@as(u64, 2), climbingStairs(2));
}

test "climbing stairs: known values" {
    try testing.expectEqual(@as(u64, 3), climbingStairs(3));
    try testing.expectEqual(@as(u64, 5), climbingStairs(4));
    try testing.expectEqual(@as(u64, 8), climbingStairs(5));
}

test "climbing stairs: larger value" {
    try testing.expectEqual(@as(u64, 89), climbingStairs(10));
}
