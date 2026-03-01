//! Fibonacci (DP memoization) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/fibonacci.py

const std = @import("std");
const testing = std.testing;

pub const FibonacciError = error{Overflow};

/// Returns the n-th Fibonacci number (0-indexed) using top-down memoization.
/// Time complexity: O(n), Space complexity: O(n)
pub fn fibonacciDp(allocator: std.mem.Allocator, n: usize) (FibonacciError || std.mem.Allocator.Error)![]u64 {
    const n_plus = @addWithOverflow(n, @as(usize, 1));
    if (n_plus[1] != 0) return FibonacciError.Overflow;

    const memo = try allocator.alloc(u64, n_plus[0]);
    defer allocator.free(memo);

    const unseen = std.math.maxInt(u64);
    @memset(memo, unseen);

    memo[0] = 0;
    if (n >= 1) memo[1] = 1;

    const sequence = try allocator.alloc(u64, n_plus[0]);
    errdefer allocator.free(sequence);
    for (0..n_plus[0]) |i| {
        sequence[i] = try fibMemo(i, memo, unseen);
    }
    return sequence;
}

fn fibMemo(n: usize, memo: []u64, unseen: u64) FibonacciError!u64 {
    if (memo[n] != unseen) return memo[n];
    const left = try fibMemo(n - 1, memo, unseen);
    const right = try fibMemo(n - 2, memo, unseen);
    const sum = @addWithOverflow(left, right);
    if (sum[1] != 0) return FibonacciError.Overflow;
    memo[n] = sum[0];
    return memo[n];
}

test "fibonacci dp: base cases" {
    const alloc = testing.allocator;
    const seq0 = try fibonacciDp(alloc, 0);
    defer alloc.free(seq0);
    try testing.expectEqualSlices(u64, &[_]u64{0}, seq0);

    const seq1 = try fibonacciDp(alloc, 1);
    defer alloc.free(seq1);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1 }, seq1);
}

test "fibonacci dp: known values" {
    const alloc = testing.allocator;
    const seq = try fibonacciDp(alloc, 10);
    defer alloc.free(seq);
    try testing.expectEqualSlices(
        u64,
        &[_]u64{ 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 },
        seq,
    );
}

test "fibonacci dp: larger value" {
    const alloc = testing.allocator;
    const seq = try fibonacciDp(alloc, 20);
    defer alloc.free(seq);
    try testing.expectEqual(@as(u64, 6765), seq[20]);
}

test "fibonacci dp: overflow is reported" {
    try testing.expectError(FibonacciError.Overflow, fibonacciDp(testing.allocator, 94));
}
