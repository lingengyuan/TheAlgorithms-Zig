//! Is Even - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/is_even.py

const std = @import("std");
const testing = std.testing;

/// Returns true when the integer is even.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn isEven(number: i64) bool {
    return (number & 1) == 0;
}

test "is even: python examples" {
    try testing.expect(!isEven(1));
    try testing.expect(isEven(4));
    try testing.expect(!isEven(9));
    try testing.expect(!isEven(15));
    try testing.expect(isEven(40));
    try testing.expect(isEven(100));
    try testing.expect(!isEven(101));
}

test "is even: edge and extreme" {
    try testing.expect(isEven(0));
    try testing.expect(isEven(-2));
    try testing.expect(!isEven(std.math.minInt(i64) + 1));
    try testing.expect(isEven(std.math.minInt(i64)));
}
