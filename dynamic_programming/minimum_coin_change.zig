//! Minimum Coin Change (compatibility wrapper) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_coin_change.py

const std = @import("std");
const testing = std.testing;
const existing = @import("coin_change.zig");

pub const CoinChangeError = existing.CoinChangeError;

/// Returns the number of ways to make `amount` using unlimited coins.
/// Time complexity: O(amount * coin_count), Space complexity: O(amount)
pub fn minimumCoinChange(
    allocator: std.mem.Allocator,
    coins: []const u32,
    amount: i32,
) (CoinChangeError || std.mem.Allocator.Error)!u64 {
    return existing.coinChangeWays(allocator, coins, amount);
}

test "minimum coin change: python examples" {
    try testing.expectEqual(@as(u64, 4), try minimumCoinChange(testing.allocator, &[_]u32{ 1, 2, 3 }, 4));
    try testing.expectEqual(@as(u64, 5), try minimumCoinChange(testing.allocator, &[_]u32{ 2, 5, 3, 6 }, 10));
}

test "minimum coin change: boundaries" {
    try testing.expectEqual(@as(u64, 1), try minimumCoinChange(testing.allocator, &[_]u32{ 4, 5, 6 }, 0));
    try testing.expectEqual(@as(u64, 0), try minimumCoinChange(testing.allocator, &[_]u32{10}, 99));
    try testing.expectEqual(@as(u64, 0), try minimumCoinChange(testing.allocator, &[_]u32{ 1, 2, 3 }, -5));
}

test "minimum coin change: extreme overflow" {
    var many_ones: [65]u32 = undefined;
    @memset(&many_ones, 1);
    try testing.expectError(CoinChangeError.Overflow, minimumCoinChange(testing.allocator, &many_ones, 64));
}
