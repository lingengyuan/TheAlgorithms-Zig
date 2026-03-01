//! Coin Change (Number of Ways) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_coin_change.py

const std = @import("std");
const testing = std.testing;

pub const CoinChangeError = error{Overflow};

/// Returns the number of ways to make `amount` using unlimited coins.
/// For negative amounts, returns 0.
/// Time complexity: O(amount * coin_count), Space complexity: O(amount)
pub fn coinChangeWays(
    allocator: std.mem.Allocator,
    coins: []const u32,
    amount: i32,
) (CoinChangeError || std.mem.Allocator.Error)!u64 {
    if (amount < 0) return 0;
    if (amount == 0) return 1;

    const target: usize = @intCast(amount);
    const size = @addWithOverflow(target, @as(usize, 1));
    if (size[1] != 0) return CoinChangeError.Overflow;
    const dp = try allocator.alloc(u64, size[0]);
    defer allocator.free(dp);

    @memset(dp, 0);
    dp[0] = 1;

    for (coins) |coin| {
        var value = coin;
        while (value <= @as(u32, @intCast(amount))) : (value += 1) {
            const idx: usize = @intCast(value);
            const prev_idx: usize = @intCast(value - coin);
            const next = @addWithOverflow(dp[idx], dp[prev_idx]);
            if (next[1] != 0) return CoinChangeError.Overflow;
            dp[idx] = next[0];
        }
    }

    return dp[target];
}

test "coin change: basic case" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 4), try coinChangeWays(alloc, &[_]u32{ 1, 2, 3 }, 4));
}

test "coin change: known values" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 8), try coinChangeWays(alloc, &[_]u32{ 1, 2, 3 }, 7));
    try testing.expectEqual(@as(u64, 5), try coinChangeWays(alloc, &[_]u32{ 2, 5, 3, 6 }, 10));
}

test "coin change: zero amount" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 1), try coinChangeWays(alloc, &[_]u32{ 4, 5, 6 }, 0));
}

test "coin change: impossible case" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 0), try coinChangeWays(alloc, &[_]u32{10}, 99));
}

test "coin change: negative amount" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 0), try coinChangeWays(alloc, &[_]u32{ 1, 2, 3 }, -5));
}

test "coin change: combinatorial overflow is reported" {
    var many_ones: [65]u32 = undefined;
    @memset(&many_ones, 1);
    try testing.expectError(CoinChangeError.Overflow, coinChangeWays(testing.allocator, &many_ones, 64));
}
