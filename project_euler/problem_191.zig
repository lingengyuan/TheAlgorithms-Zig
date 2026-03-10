//! Project Euler Problem 191: Prize Strings - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_191/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of valid prize strings over `days` days.
/// Time complexity: O(days)
/// Space complexity: O(1)
pub fn solution(days: u32) u64 {
    var dp = [_][3]u64{
        .{ 1, 0, 0 },
        .{ 0, 0, 0 },
    };

    var day: u32 = 0;
    while (day < days) : (day += 1) {
        var next = [_][3]u64{
            .{ 0, 0, 0 },
            .{ 0, 0, 0 },
        };

        for (0..2) |late_used| {
            for (0..3) |absent_streak| {
                const ways = dp[late_used][absent_streak];
                if (ways == 0) continue;
                next[late_used][0] += ways;
                if (absent_streak < 2) next[late_used][absent_streak + 1] += ways;
                if (late_used == 0) next[1][0] += ways;
            }
        }
        dp = next;
    }

    var total: u64 = 0;
    for (dp) |row| {
        for (row) |value| total += value;
    }
    return total;
}

test "problem 191: python reference" {
    try testing.expectEqual(@as(u64, 43), solution(4));
    try testing.expectEqual(@as(u64, 1918080160), solution(30));
}

test "problem 191: small day counts" {
    try testing.expectEqual(@as(u64, 1), solution(0));
    try testing.expectEqual(@as(u64, 3), solution(1));
    try testing.expectEqual(@as(u64, 8), solution(2));
}
