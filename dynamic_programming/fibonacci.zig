//! Fibonacci (DP compatibility wrapper) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/fibonacci.py

const std = @import("std");
const testing = std.testing;
const existing = @import("fibonacci_dp.zig");

pub const FibonacciError = existing.FibonacciError;

/// Returns the Fibonacci sequence from `0` through `n` inclusive.
/// Time complexity: O(n), Space complexity: O(n)
pub fn fibonacci(allocator: std.mem.Allocator, n: usize) (FibonacciError || std.mem.Allocator.Error)![]u64 {
    return existing.fibonacciDp(allocator, n);
}

test "dynamic programming fibonacci: base cases" {
    const seq0 = try fibonacci(testing.allocator, 0);
    defer testing.allocator.free(seq0);
    try testing.expectEqualSlices(u64, &[_]u64{0}, seq0);

    const seq1 = try fibonacci(testing.allocator, 1);
    defer testing.allocator.free(seq1);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1 }, seq1);
}

test "dynamic programming fibonacci: known values" {
    const seq = try fibonacci(testing.allocator, 10);
    defer testing.allocator.free(seq);
    try testing.expectEqual(@as(u64, 55), seq[10]);
}

test "dynamic programming fibonacci: extreme overflow" {
    try testing.expectError(FibonacciError.Overflow, fibonacci(testing.allocator, 94));
}
