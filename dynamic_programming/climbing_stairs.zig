//! Climbing Stairs - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/climbing_stairs.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of distinct ways to climb `n` steps
/// when each move can be either 1 or 2 steps.
/// Time complexity: O(n), Space complexity: O(1)
pub fn climbingStairs(number_of_steps: i32) !u64 {
    if (number_of_steps <= 0) return error.InvalidInput;
    if (number_of_steps == 1) return 1;

    var prev2: u64 = 1;
    var prev1: u64 = 1;

    var i: i32 = 0;
    while (i < number_of_steps - 1) : (i += 1) {
        const current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    return prev1;
}

test "climbing stairs: base cases" {
    try testing.expectEqual(@as(u64, 1), try climbingStairs(1));
    try testing.expectEqual(@as(u64, 2), try climbingStairs(2));
}

test "climbing stairs: known values" {
    try testing.expectEqual(@as(u64, 3), try climbingStairs(3));
    try testing.expectEqual(@as(u64, 5), try climbingStairs(4));
    try testing.expectEqual(@as(u64, 8), try climbingStairs(5));
}

test "climbing stairs: larger value" {
    try testing.expectEqual(@as(u64, 89), try climbingStairs(10));
}

test "climbing stairs: invalid input" {
    try testing.expectError(error.InvalidInput, climbingStairs(0));
    try testing.expectError(error.InvalidInput, climbingStairs(-7));
}
