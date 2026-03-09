//! Project Euler Problem 38: Pandigital Multiples - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_038/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns whether `n` is a 9-digit 1-to-9 pandigital integer.
///
/// Time complexity: O(9)
/// Space complexity: O(1)
pub fn is9Pandigital(n: u64) bool {
    var value = n;
    var seen: u16 = 0;
    var digits: u8 = 0;

    while (value > 0) : (digits += 1) {
        const digit: u8 = @intCast(value % 10);
        if (digit == 0) return false;
        const mask: u16 = @as(u16, 1) << @as(u4, @intCast(digit));
        if ((seen & mask) != 0) return false;
        seen |= mask;
        value /= 10;
    }

    return digits == 9 and seen == 0b11_1111_1110;
}

/// Returns the largest pandigital concatenated product, or `null` if none exists.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn solution() ?u64 {
    var base_num: u64 = 9_999;
    while (base_num >= 5_000) : (base_num -= 1) {
        const candidate = 100_002 * base_num;
        if (is9Pandigital(candidate)) return candidate;
    }

    base_num = 333;
    while (base_num >= 100) : (base_num -= 1) {
        const candidate = 1_002_003 * base_num;
        if (is9Pandigital(candidate)) return candidate;
    }

    return null;
}

test "problem 038: python reference" {
    try testing.expectEqual(@as(?u64, 932_718_654), solution());
}

test "problem 038: helper semantics and extremes" {
    try testing.expect(!is9Pandigital(12_345));
    try testing.expect(is9Pandigital(156_284_973));
    try testing.expect(!is9Pandigital(1_562_849_733));
    try testing.expect(!is9Pandigital(918_273_640));
}
