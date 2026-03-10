//! Project Euler Problem 109: Darts - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_109/sol1.py

const std = @import("std");
const testing = std.testing;

fn allThrows() [63]u32 {
    var values: [63]u32 = undefined;
    var idx: usize = 0;
    for (1..21) |n| {
        values[idx] = @intCast(n);
        idx += 1;
    }
    values[idx] = 25;
    idx += 1;
    for (1..21) |n| {
        values[idx] = @as(u32, @intCast(2 * n));
        idx += 1;
    }
    values[idx] = 50;
    idx += 1;
    for (1..21) |n| {
        values[idx] = @as(u32, @intCast(3 * n));
        idx += 1;
    }
    values[idx] = 0;
    return values;
}

fn finishingDoubles() [21]u32 {
    var values: [21]u32 = undefined;
    for (1..21) |n| values[n - 1] = @as(u32, @intCast(2 * n));
    values[20] = 50;
    return values;
}

/// Returns the number of distinct checkout combinations with score less than `limit`.
/// Time complexity: O(doubles * throws^2)
/// Space complexity: O(1)
pub fn solution(limit: u32) u32 {
    if (limit <= 1) return 0;

    const throws = allThrows();
    const doubles = finishingDoubles();

    var checkouts: u32 = 0;
    for (doubles) |double| {
        for (0..throws.len) |i| {
            for (i..throws.len) |j| {
                if (double + throws[i] + throws[j] < limit) checkouts += 1;
            }
        }
    }
    return checkouts;
}

test "problem 109: python reference" {
    try testing.expectEqual(@as(u32, 38182), solution(100));
    try testing.expectEqual(@as(u32, 42336), solution(171));
}

test "problem 109: smaller limits and extremes" {
    try testing.expectEqual(@as(u32, 12577), solution(50));
    try testing.expectEqual(@as(u32, 1), solution(3));
    try testing.expectEqual(@as(u32, 0), solution(1));
}
