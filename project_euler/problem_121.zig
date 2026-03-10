//! Project Euler Problem 121: Disc Game Prize Fund - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_121/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the maximum prize fund that avoids an expected loss after `num_turns` turns.
/// Time complexity: O(num_turns^2)
/// Space complexity: O(num_turns)
pub fn solution(num_turns: u32) u64 {
    var dp: [64]u128 = [_]u128{0} ** 64;
    dp[0] = 1;

    var turn: u32 = 0;
    while (turn < num_turns) : (turn += 1) {
        const total_discs = @as(u128, turn) + 2;
        var blue_count = turn + 1;
        while (blue_count > 0) : (blue_count -= 1) {
            dp[blue_count] = dp[blue_count] * (total_discs - 1) + dp[blue_count - 1];
        }
        dp[0] *= total_discs - 1;
    }

    var winning_numerator: u128 = 0;
    var k = num_turns / 2 + 1;
    while (k <= num_turns) : (k += 1) winning_numerator += dp[k];

    var denominator: u128 = 1;
    var discs: u128 = 2;
    while (discs <= num_turns + 1) : (discs += 1) denominator *= discs;

    return @intCast(denominator / winning_numerator);
}

test "problem 121: python reference" {
    try testing.expectEqual(@as(u64, 10), solution(4));
    try testing.expectEqual(@as(u64, 2269), solution(15));
}

test "problem 121: short games" {
    try testing.expectEqual(@as(u64, 2), solution(1));
    try testing.expectEqual(@as(u64, 6), solution(2));
}
