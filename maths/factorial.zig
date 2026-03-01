//! Factorial - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/factorial.py

const std = @import("std");
const testing = std.testing;

pub const FactorialError = error{Overflow};

/// Computes n! iteratively.
/// Time complexity: O(n), Space complexity: O(1)
pub fn factorial(n: u32) FactorialError!u64 {
    var result: u64 = 1;
    for (1..@as(u64, n) + 1) |i| {
        const with_overflow = @mulWithOverflow(result, i);
        if (with_overflow[1] != 0) return FactorialError.Overflow;
        result = with_overflow[0];
    }
    return result;
}

test "factorial: base cases" {
    try testing.expectEqual(@as(u64, 1), try factorial(0));
    try testing.expectEqual(@as(u64, 1), try factorial(1));
}

test "factorial: known values" {
    try testing.expectEqual(@as(u64, 2), try factorial(2));
    try testing.expectEqual(@as(u64, 6), try factorial(3));
    try testing.expectEqual(@as(u64, 24), try factorial(4));
    try testing.expectEqual(@as(u64, 120), try factorial(5));
    try testing.expectEqual(@as(u64, 720), try factorial(6));
}

test "factorial: larger values" {
    try testing.expectEqual(@as(u64, 3628800), try factorial(10));
    try testing.expectEqual(@as(u64, 2432902008176640000), try factorial(20));
}

test "factorial: overflow is reported" {
    try testing.expectError(FactorialError.Overflow, factorial(21));
}
