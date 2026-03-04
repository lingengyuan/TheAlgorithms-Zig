//! Double Factorial - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/double_factorial.py

const std = @import("std");
const testing = std.testing;

pub const DoubleFactorialError = error{ InvalidInput, Overflow };

/// Computes double factorial recursively.
/// Time complexity: O(n), Space complexity: O(n)
pub fn doubleFactorialRecursive(n: i64) DoubleFactorialError!u128 {
    if (n < 0) return DoubleFactorialError.InvalidInput;
    if (n <= 1) return 1;

    const prev = try doubleFactorialRecursive(n - 2);
    const mul = @mulWithOverflow(@as(u128, @intCast(n)), prev);
    if (mul[1] != 0) return DoubleFactorialError.Overflow;
    return mul[0];
}

/// Computes double factorial iteratively.
/// Time complexity: O(n), Space complexity: O(1)
pub fn doubleFactorialIterative(num: i64) DoubleFactorialError!u128 {
    if (num < 0) return DoubleFactorialError.InvalidInput;

    var value: u128 = 1;
    var i: i64 = num;
    while (i > 0) : (i -= 2) {
        const mul = @mulWithOverflow(value, @as(u128, @intCast(i)));
        if (mul[1] != 0) return DoubleFactorialError.Overflow;
        value = mul[0];
    }
    return value;
}

test "double factorial: python reference examples" {
    var i: i64 = 0;
    while (i < 20) : (i += 1) {
        try testing.expectEqual(try doubleFactorialIterative(i), try doubleFactorialRecursive(i));
    }

    try testing.expectError(DoubleFactorialError.InvalidInput, doubleFactorialRecursive(-1));
    try testing.expectError(DoubleFactorialError.InvalidInput, doubleFactorialIterative(-1));
}

test "double factorial: edge and extreme cases" {
    try testing.expectEqual(@as(u128, 1), try doubleFactorialIterative(0));
    try testing.expectEqual(@as(u128, 1), try doubleFactorialIterative(1));
    try testing.expectEqual(@as(u128, 3_715_891_200), try doubleFactorialIterative(20));
    try testing.expectError(DoubleFactorialError.Overflow, doubleFactorialIterative(200));
}
