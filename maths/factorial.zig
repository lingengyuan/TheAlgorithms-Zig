//! Factorial - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/factorial.py

const std = @import("std");
const testing = std.testing;

/// Computes n! iteratively.
/// Time complexity: O(n), Space complexity: O(1)
pub fn factorial(n: u32) u64 {
    var result: u64 = 1;
    for (1..@as(u64, n) + 1) |i| {
        result *= i;
    }
    return result;
}

test "factorial: base cases" {
    try testing.expectEqual(@as(u64, 1), factorial(0));
    try testing.expectEqual(@as(u64, 1), factorial(1));
}

test "factorial: known values" {
    try testing.expectEqual(@as(u64, 2), factorial(2));
    try testing.expectEqual(@as(u64, 6), factorial(3));
    try testing.expectEqual(@as(u64, 24), factorial(4));
    try testing.expectEqual(@as(u64, 120), factorial(5));
    try testing.expectEqual(@as(u64, 720), factorial(6));
}

test "factorial: larger values" {
    try testing.expectEqual(@as(u64, 3628800), factorial(10));
    try testing.expectEqual(@as(u64, 2432902008176640000), factorial(20));
}
