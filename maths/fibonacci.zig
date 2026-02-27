//! Fibonacci - Zig implementation (iterative)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/fibonacci.py

const std = @import("std");
const testing = std.testing;

/// Returns the n-th Fibonacci number (0-indexed).
/// fib(0) = 0, fib(1) = 1, fib(2) = 1, fib(3) = 2, ...
/// Time complexity: O(n), Space complexity: O(1)
pub fn fibonacci(n: u32) u64 {
    if (n == 0) return 0;
    if (n == 1) return 1;
    var a: u64 = 0;
    var b: u64 = 1;
    for (2..n + 1) |_| {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

test "fibonacci: base cases" {
    try testing.expectEqual(@as(u64, 0), fibonacci(0));
    try testing.expectEqual(@as(u64, 1), fibonacci(1));
    try testing.expectEqual(@as(u64, 1), fibonacci(2));
}

test "fibonacci: known values" {
    try testing.expectEqual(@as(u64, 5), fibonacci(5));
    try testing.expectEqual(@as(u64, 55), fibonacci(10));
    try testing.expectEqual(@as(u64, 6765), fibonacci(20));
}

test "fibonacci: larger value" {
    try testing.expectEqual(@as(u64, 832040), fibonacci(30));
}
