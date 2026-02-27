//! Least Common Multiple - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/least_common_multiple.py

const std = @import("std");
const testing = std.testing;

/// LCM via GCD: lcm(a, b) = |a| / gcd(a, b) * |b|
/// Time complexity: O(log(min(a, b))), Space complexity: O(1)
pub fn lcm(a: i64, b: i64) u64 {
    if (a == 0 or b == 0) return 0;
    const abs_a = if (a < 0) @as(u64, @intCast(-a)) else @as(u64, @intCast(a));
    const abs_b = if (b < 0) @as(u64, @intCast(-b)) else @as(u64, @intCast(b));
    return abs_a / gcd_internal(abs_a, abs_b) * abs_b;
}

fn gcd_internal(a: u64, b: u64) u64 {
    var x = a;
    var y = b;
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

test "lcm: basic cases" {
    try testing.expectEqual(@as(u64, 10), lcm(5, 2));
    try testing.expectEqual(@as(u64, 228), lcm(12, 76));
    try testing.expectEqual(@as(u64, 20), lcm(10, 20));
    try testing.expectEqual(@as(u64, 195), lcm(13, 15));
    try testing.expectEqual(@as(u64, 124), lcm(4, 31));
}

test "lcm: with zero" {
    try testing.expectEqual(@as(u64, 0), lcm(0, 5));
    try testing.expectEqual(@as(u64, 0), lcm(7, 0));
}

test "lcm: same number" {
    try testing.expectEqual(@as(u64, 6), lcm(6, 6));
}

test "lcm: coprime" {
    try testing.expectEqual(@as(u64, 35), lcm(5, 7));
}
