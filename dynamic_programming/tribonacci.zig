//! Tribonacci Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/tribonacci.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const TribonacciError = error{
    InvalidInput,
    Overflow,
};

/// Returns the first `num` tribonacci numbers, matching Python behavior:
/// sequence starts with [0, 0, 1, ...], and `num` must be at least 3.
/// Time complexity: O(num), Space complexity: O(num)
pub fn tribonacci(
    allocator: Allocator,
    num: usize,
) (TribonacciError || Allocator.Error)![]u64 {
    if (num < 3) return TribonacciError.InvalidInput;

    const dp = try allocator.alloc(u64, num);
    errdefer allocator.free(dp);

    @memset(dp, 0);
    dp[2] = 1;

    for (3..num) |i| {
        const first_sum = @addWithOverflow(dp[i - 1], dp[i - 2]);
        if (first_sum[1] != 0) return TribonacciError.Overflow;
        const second_sum = @addWithOverflow(first_sum[0], dp[i - 3]);
        if (second_sum[1] != 0) return TribonacciError.Overflow;
        dp[i] = second_sum[0];
    }

    return dp;
}

test "tribonacci: python examples" {
    const seq5 = try tribonacci(testing.allocator, 5);
    defer testing.allocator.free(seq5);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 1, 1, 2 }, seq5);

    const seq8 = try tribonacci(testing.allocator, 8);
    defer testing.allocator.free(seq8);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 1, 1, 2, 4, 7, 13 }, seq8);
}

test "tribonacci: boundary and invalid input" {
    const seq3 = try tribonacci(testing.allocator, 3);
    defer testing.allocator.free(seq3);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 1 }, seq3);

    try testing.expectError(TribonacciError.InvalidInput, tribonacci(testing.allocator, 2));
    try testing.expectError(TribonacciError.InvalidInput, tribonacci(testing.allocator, 0));
}

test "tribonacci: extreme overflow detection" {
    const seq76 = try tribonacci(testing.allocator, 76);
    defer testing.allocator.free(seq76);
    try testing.expectEqual(@as(u64, 12903063846126135669), seq76[75]);

    try testing.expectError(TribonacciError.Overflow, tribonacci(testing.allocator, 77));
}
