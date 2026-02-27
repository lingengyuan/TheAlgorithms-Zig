//! Fibonacci (DP memoization) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/fibonacci.py

const std = @import("std");
const testing = std.testing;

/// Returns the n-th Fibonacci number (0-indexed) using top-down memoization.
/// Time complexity: O(n), Space complexity: O(n)
pub fn fibonacciDp(allocator: std.mem.Allocator, n: usize) !u64 {
    const memo = try allocator.alloc(u64, n + 1);
    defer allocator.free(memo);

    const unseen = std.math.maxInt(u64);
    @memset(memo, unseen);

    memo[0] = 0;
    if (n >= 1) memo[1] = 1;

    return fibMemo(n, memo, unseen);
}

fn fibMemo(n: usize, memo: []u64, unseen: u64) u64 {
    if (memo[n] != unseen) return memo[n];
    memo[n] = fibMemo(n - 1, memo, unseen) + fibMemo(n - 2, memo, unseen);
    return memo[n];
}

test "fibonacci dp: base cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 0), try fibonacciDp(alloc, 0));
    try testing.expectEqual(@as(u64, 1), try fibonacciDp(alloc, 1));
}

test "fibonacci dp: known values" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 1), try fibonacciDp(alloc, 2));
    try testing.expectEqual(@as(u64, 2), try fibonacciDp(alloc, 3));
    try testing.expectEqual(@as(u64, 55), try fibonacciDp(alloc, 10));
}

test "fibonacci dp: larger value" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 6765), try fibonacciDp(alloc, 20));
}
