//! Numbers Different Signs - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/numbers_different_signs.py

const std = @import("std");
const testing = std.testing;

/// Returns true when `num1` and `num2` have opposite signs.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn differentSigns(num1: i64, num2: i64) bool {
    return (num1 ^ num2) < 0;
}

test "numbers different signs: python examples" {
    try testing.expect(differentSigns(1, -1));
    try testing.expect(!differentSigns(1, 1));
    try testing.expect(differentSigns(std.math.maxInt(i64), std.math.minInt(i64) + 1));
    try testing.expect(!differentSigns(50, 278));
    try testing.expect(!differentSigns(0, 2));
    try testing.expect(!differentSigns(2, 0));
}

test "numbers different signs: edge cases" {
    try testing.expect(!differentSigns(0, 0));
    try testing.expect(differentSigns(-1, 0x7FFF_FFFF));
    try testing.expect(!differentSigns(std.math.minInt(i64), -1));
}
