//! Rod Cutting - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/rod_cutting.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the maximum obtainable value for a rod of `length`.
/// `prices[i]` is the price of a rod segment with length `i + 1`.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn rodCutting(allocator: Allocator, prices: []const i64, length: usize) !i64 {
    if (length == 0) return 0;

    const dp = try allocator.alloc(i64, length + 1);
    defer allocator.free(dp);
    @memset(dp, 0);

    const min_i64: i64 = std.math.minInt(i64);
    for (1..length + 1) |rod_len| {
        var best: i64 = min_i64;
        for (1..rod_len + 1) |cut| {
            if (cut > prices.len) continue;
            const sum = @addWithOverflow(prices[cut - 1], dp[rod_len - cut]);
            if (sum[1] != 0) continue;
            if (sum[0] > best) best = sum[0];
        }
        dp[rod_len] = if (best == min_i64) 0 else best;
    }

    return dp[length];
}

test "rod cutting: classic example length 8" {
    const alloc = testing.allocator;
    const prices = [_]i64{ 1, 5, 8, 9, 10, 17, 17, 20 };
    try testing.expectEqual(@as(i64, 22), try rodCutting(alloc, &prices, 8));
}

test "rod cutting: classic example length 4" {
    const alloc = testing.allocator;
    const prices = [_]i64{ 1, 5, 8, 9, 10, 17, 17, 20 };
    try testing.expectEqual(@as(i64, 10), try rodCutting(alloc, &prices, 4));
}

test "rod cutting: zero length" {
    const alloc = testing.allocator;
    const prices = [_]i64{ 2, 5, 7 };
    try testing.expectEqual(@as(i64, 0), try rodCutting(alloc, &prices, 0));
}

test "rod cutting: empty prices" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(i64, 0), try rodCutting(alloc, &[_]i64{}, 5));
}

test "rod cutting: direct no-cut is optimal" {
    const alloc = testing.allocator;
    const prices = [_]i64{ 2, 5, 9, 20 };
    try testing.expectEqual(@as(i64, 20), try rodCutting(alloc, &prices, 4));
}
