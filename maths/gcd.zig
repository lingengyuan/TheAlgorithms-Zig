//! Greatest Common Divisor (Euclidean algorithm) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/greatest_common_divisor.py

const std = @import("std");
const testing = std.testing;

/// Iterative Euclidean algorithm for GCD.
/// Time complexity: O(log(min(a, b))), Space complexity: O(1)
pub fn gcd(a: i64, b: i64) u64 {
    var x = if (a < 0) @as(u64, @intCast(-a)) else @as(u64, @intCast(a));
    var y = if (b < 0) @as(u64, @intCast(-b)) else @as(u64, @intCast(b));
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

test "gcd: basic cases" {
    try testing.expectEqual(@as(u64, 8), gcd(24, 40));
    try testing.expectEqual(@as(u64, 1), gcd(1, 1));
    try testing.expectEqual(@as(u64, 1), gcd(1, 800));
    try testing.expectEqual(@as(u64, 1), gcd(11, 37));
    try testing.expectEqual(@as(u64, 4), gcd(16, 4));
}

test "gcd: negative numbers" {
    try testing.expectEqual(@as(u64, 3), gcd(-3, 9));
    try testing.expectEqual(@as(u64, 3), gcd(9, -3));
    try testing.expectEqual(@as(u64, 3), gcd(3, -9));
    try testing.expectEqual(@as(u64, 3), gcd(-3, -9));
}

test "gcd: zero" {
    try testing.expectEqual(@as(u64, 5), gcd(0, 5));
    try testing.expectEqual(@as(u64, 7), gcd(7, 0));
    try testing.expectEqual(@as(u64, 0), gcd(0, 0));
}
