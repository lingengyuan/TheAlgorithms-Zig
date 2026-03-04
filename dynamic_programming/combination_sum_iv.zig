//! Combination Sum IV (Ordered Combinations) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/combination_sum_iv.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CombinationSumError = error{
    InvalidInput,
    Overflow,
};

/// Returns the number of ordered combinations of `nums` that sum to `target`.
/// Equivalent to Python `combination_sum_iv_bottom_up`.
/// Time complexity: O(target * nums.len), Space complexity: O(target)
pub fn combinationSumIv(
    allocator: Allocator,
    nums: []const i32,
    target: i32,
) (CombinationSumError || Allocator.Error)!u64 {
    if (target < 0) return 0;
    if (target == 0) return 1;

    for (nums) |num| {
        if (num <= 0) return CombinationSumError.InvalidInput;
    }

    const target_usize: usize = @intCast(target);
    const len_with_zero = @addWithOverflow(target_usize, @as(usize, 1));
    if (len_with_zero[1] != 0) return CombinationSumError.Overflow;

    const dp = try allocator.alloc(u64, len_with_zero[0]);
    defer allocator.free(dp);
    @memset(dp, 0);
    dp[0] = 1;

    for (1..len_with_zero[0]) |sum_idx| {
        for (nums) |num| {
            const num_usize: usize = @intCast(num);
            if (num_usize > sum_idx) continue;
            const addend = dp[sum_idx - num_usize];
            const next = @addWithOverflow(dp[sum_idx], addend);
            if (next[1] != 0) return CombinationSumError.Overflow;
            dp[sum_idx] = next[0];
        }
    }

    return dp[target_usize];
}

test "combination sum iv: python sample" {
    const nums = [_]i32{ 1, 2, 5 };
    try testing.expectEqual(@as(u64, 9), try combinationSumIv(testing.allocator, &nums, 5));
}

test "combination sum iv: base and impossible cases" {
    const nums = [_]i32{ 2, 4 };
    try testing.expectEqual(@as(u64, 1), try combinationSumIv(testing.allocator, &nums, 0));
    try testing.expectEqual(@as(u64, 0), try combinationSumIv(testing.allocator, &nums, 7));
    try testing.expectEqual(@as(u64, 0), try combinationSumIv(testing.allocator, &nums, -3));
}

test "combination sum iv: ordered combinations differ from coin-change combinations" {
    const nums = [_]i32{ 1, 2 };
    // Ordered sequences for 4: 1111, 112, 121, 211, 22 => 5
    try testing.expectEqual(@as(u64, 5), try combinationSumIv(testing.allocator, &nums, 4));
}

test "combination sum iv: invalid numbers" {
    const nums_with_zero = [_]i32{ 0, 1 };
    try testing.expectError(CombinationSumError.InvalidInput, combinationSumIv(testing.allocator, &nums_with_zero, 2));

    const nums_with_negative = [_]i32{ -1, 2 };
    try testing.expectError(CombinationSumError.InvalidInput, combinationSumIv(testing.allocator, &nums_with_negative, 2));
}

test "combination sum iv: extreme overflow detection" {
    const nums = [_]i32{ 1, 2 };
    // Count grows as Fibonacci and exceeds u64 at this scale.
    try testing.expectError(CombinationSumError.Overflow, combinationSumIv(testing.allocator, &nums, 100));
}
