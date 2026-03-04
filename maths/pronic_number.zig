//! Pronic Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/pronic_number.py

const std = @import("std");
const testing = std.testing;

/// Returns true when `number = m * (m + 1)` for some integer `m`.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn isPronic(number: i64) bool {
    if (number < 0 or @mod(number, @as(i64, 2)) == 1) return false;

    const n: u128 = @intCast(number);
    const root = integerSqrt(n);
    return n == root * (root + 1);
}

fn integerSqrt(value: u128) u128 {
    if (value < 2) return value;
    var x = value;
    var y = (x + value / x) / 2;
    while (y < x) {
        x = y;
        y = (x + value / x) / 2;
    }
    return x;
}

test "pronic number: python reference examples" {
    try testing.expect(!isPronic(-1));
    try testing.expect(isPronic(0));
    try testing.expect(isPronic(2));
    try testing.expect(!isPronic(5));
    try testing.expect(isPronic(6));
    try testing.expect(!isPronic(8));
    try testing.expect(isPronic(30));
    try testing.expect(!isPronic(32));
    try testing.expect(isPronic(2_147_441_940));
    try testing.expect(isPronic(9_223_372_033_963_249_500));
}

test "pronic number: extreme boundary" {
    try testing.expect(!isPronic(std.math.maxInt(i64)));
}
