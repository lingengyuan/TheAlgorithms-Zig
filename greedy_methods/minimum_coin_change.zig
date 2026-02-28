//! Minimum Coin Change (Greedy) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/greedy_methods/minimum_coin_change.py
//! Note: greedy only works correctly for canonical coin systems (e.g. standard currency).

const std = @import("std");
const testing = std.testing;

/// Returns the minimum list of coins (greedy, largest-first) that sum to `amount`.
/// Coins must be sorted in descending order. Caller owns the returned slice.
pub fn minimumCoinChange(allocator: std.mem.Allocator, coins: []const u64, amount: u64) ![]u64 {
    var result = std.ArrayListUnmanaged(u64){};
    defer result.deinit(allocator);
    var remaining = amount;
    for (coins) |coin| {
        while (remaining >= coin) {
            try result.append(allocator, coin);
            remaining -= coin;
        }
    }
    const out = try allocator.alloc(u64, result.items.len);
    @memcpy(out, result.items);
    return out;
}

test "minimum coin: without 200 denomination" {
    const alloc = testing.allocator;
    const coins = [_]u64{ 500, 100, 50, 20, 10, 5, 2, 1 };
    const result = try minimumCoinChange(alloc, &coins, 987);
    defer alloc.free(result);
    try testing.expectEqualSlices(u64, &[_]u64{ 500, 100, 100, 100, 100, 50, 20, 10, 5, 2 }, result);
}

test "minimum coin: with 200 picks fewer coins" {
    const alloc = testing.allocator;
    const coins = [_]u64{ 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1 };
    const result = try minimumCoinChange(alloc, &coins, 987);
    defer alloc.free(result);
    // 500+200+200+50+20+10+5+2 = 987
    try testing.expectEqualSlices(u64, &[_]u64{ 500, 200, 200, 50, 20, 10, 5, 2 }, result);
}

test "minimum coin: zero amount" {
    const alloc = testing.allocator;
    const coins = [_]u64{ 100, 50, 25, 10, 5, 1 };
    const result = try minimumCoinChange(alloc, &coins, 0);
    defer alloc.free(result);
    try testing.expectEqual(@as(usize, 0), result.len);
}

test "minimum coin: exact single coin" {
    const alloc = testing.allocator;
    const coins = [_]u64{ 100, 50, 25, 10, 5, 1 };
    const result = try minimumCoinChange(alloc, &coins, 100);
    defer alloc.free(result);
    try testing.expectEqualSlices(u64, &[_]u64{100}, result);
}
