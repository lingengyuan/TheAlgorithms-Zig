//! Power Using Recursion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/power_using_recursion.py

const std = @import("std");
const testing = std.testing;

pub const PowerError = error{InvalidExponent};

/// Computes `base^exponent` recursively.
/// Negative exponents are rejected explicitly for boundary safety.
/// Time complexity: O(exponent), Space complexity: O(exponent)
pub fn power(base: i64, exponent: i64) PowerError!i128 {
    if (exponent < 0) return error.InvalidExponent;
    if (exponent == 0) return 1;
    return @as(i128, base) * try power(base, exponent - 1);
}

test "power using recursion: python reference examples" {
    try testing.expectEqual(@as(i128, 81), try power(3, 4));
    try testing.expectEqual(@as(i128, 1), try power(2, 0));
    try testing.expectEqual(@as(i128, 15625), try power(5, 6));
    try testing.expectEqual(@as(i128, 21914624432020321), try power(23, 12));
}

test "power using recursion: edge cases" {
    try testing.expectEqual(@as(i128, 1), try power(0, 0));
    try testing.expectEqual(@as(i128, 0), try power(0, 1));
    try testing.expectError(error.InvalidExponent, power(2, -1));
}
