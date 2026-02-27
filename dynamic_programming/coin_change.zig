//! Minimum Coin Change - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_coin_change.py

const std = @import("std");
const testing = std.testing;

/// Returns the minimum number of coins needed to make `amount`.
/// Returns null when it is impossible.
/// Time complexity: O(amount * coin_count), Space complexity: O(amount)
pub fn minCoinChange(allocator: std.mem.Allocator, coins: []const u32, amount: u32) !?u32 {
    if (amount == 0) return 0;
    if (coins.len == 0) return null;

    const size: usize = @intCast(amount + 1);
    const dp = try allocator.alloc(u32, size);
    defer allocator.free(dp);

    const inf = amount + 1;
    @memset(dp, inf);
    dp[0] = 0;

    var value: u32 = 1;
    while (value <= amount) : (value += 1) {
        for (coins) |coin| {
            if (coin <= value) {
                const prev_idx: usize = @intCast(value - coin);
                const candidate = dp[prev_idx] + 1;
                const idx: usize = @intCast(value);
                if (candidate < dp[idx]) {
                    dp[idx] = candidate;
                }
            }
        }
    }

    const ans = dp[@as(usize, @intCast(amount))];
    if (ans > amount) return null;
    return ans;
}

test "coin change: basic case" {
    const alloc = testing.allocator;
    const coins = [_]u32{ 1, 2, 5 };
    try testing.expectEqual(@as(?u32, 3), try minCoinChange(alloc, &coins, 11));
}

test "coin change: impossible case" {
    const alloc = testing.allocator;
    const coins = [_]u32{2};
    try testing.expectEqual(@as(?u32, null), try minCoinChange(alloc, &coins, 3));
}

test "coin change: zero amount" {
    const alloc = testing.allocator;
    const coins = [_]u32{ 2, 4 };
    try testing.expectEqual(@as(?u32, 0), try minCoinChange(alloc, &coins, 0));
}

test "coin change: empty coins" {
    const alloc = testing.allocator;
    const coins = [_]u32{};
    try testing.expectEqual(@as(?u32, null), try minCoinChange(alloc, &coins, 7));
}

test "coin change: exact single coin" {
    const alloc = testing.allocator;
    const coins = [_]u32{ 3, 7, 10 };
    try testing.expectEqual(@as(?u32, 1), try minCoinChange(alloc, &coins, 7));
}
