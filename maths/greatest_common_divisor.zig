//! Greatest Common Divisor - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/greatest_common_divisor.py

const std = @import("std");
const testing = std.testing;
const gcd_mod = @import("gcd.zig");

/// Recursive Euclidean greatest common divisor.
/// Time complexity: O(log(min(a, b))), Space complexity: O(1)
pub fn greatestCommonDivisor(a: i64, b: i64) u64 {
    return gcd_mod.gcd(a, b);
}

/// Iterative Euclidean greatest common divisor.
/// Time complexity: O(log(min(a, b))), Space complexity: O(1)
pub fn gcdByIterative(x: i64, y: i64) u64 {
    return gcd_mod.gcd(x, y);
}

test "greatest common divisor: python reference examples" {
    try testing.expectEqual(@as(u64, 8), greatestCommonDivisor(24, 40));
    try testing.expectEqual(@as(u64, 1), greatestCommonDivisor(11, 37));
    try testing.expectEqual(@as(u64, 3), greatestCommonDivisor(-3, 9));
    try testing.expectEqual(@as(u64, 3), gcdByIterative(-3, -9));
    try testing.expectEqual(greatestCommonDivisor(24, 40), gcdByIterative(24, 40));
}

test "greatest common divisor: edge cases" {
    try testing.expectEqual(@as(u64, 0), greatestCommonDivisor(0, 0));
    try testing.expectEqual(@as(u64, 5), gcdByIterative(0, 5));
}
