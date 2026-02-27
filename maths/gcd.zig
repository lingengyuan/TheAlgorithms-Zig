//! Greatest Common Divisor (Euclidean algorithm) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/greatest_common_divisor.py

const std = @import("std");
const testing = std.testing;

/// Iterative Euclidean algorithm for GCD.
/// Time complexity: O(log(min(a, b))), Space complexity: O(1)
pub fn gcd(a: i64, b: i64) u64 {
    var x = absI64ToU64(a);
    var y = absI64ToU64(b);
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

fn absI64ToU64(v: i64) u64 {
    const wide: i128 = v;
    const abs_wide: i128 = if (wide < 0) -wide else wide;
    return @intCast(abs_wide);
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

test "gcd: i64 min value" {
    try testing.expectEqual(@as(u64, 1), gcd(std.math.minInt(i64), 1));
    try testing.expectEqual(@as(u64, @as(u64, 1) << 63), gcd(std.math.minInt(i64), 0));
}
