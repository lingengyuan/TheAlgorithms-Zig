//! Project Euler Problem 31: Coin Sums - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_031/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem031Error = error{
    OutOfMemory,
};

const uk_coins = [8]u32{ 1, 2, 5, 10, 20, 50, 100, 200 };

/// Returns number of ways to form `n` pence with standard UK coin set.
/// Mirrors Python semantics: n < 0 yields 0.
///
/// Time complexity: O(n * number_of_coins)
/// Space complexity: O(n)
pub fn solution(n: i32, allocator: std.mem.Allocator) Problem031Error!u64 {
    if (n < 0) return 0;

    const target: usize = @intCast(n);
    var ways = try allocator.alloc(u64, target + 1);
    defer allocator.free(ways);
    @memset(ways, 0);
    ways[0] = 1;

    for (uk_coins) |coin| {
        if (coin > target) continue;

        var amount: usize = coin;
        while (amount <= target) : (amount += 1) {
            ways[amount] += ways[amount - coin];
        }
    }

    return ways[target];
}

test "problem 031: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 6_295_434), try solution(500, allocator));
    try testing.expectEqual(@as(u64, 73_682), try solution(200, allocator));
    try testing.expectEqual(@as(u64, 451), try solution(50, allocator));
    try testing.expectEqual(@as(u64, 11), try solution(10, allocator));
}

test "problem 031: boundaries and edge cases" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 0), try solution(-1, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(0, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, allocator));
    try testing.expectEqual(@as(u64, 2), try solution(2, allocator));
}
