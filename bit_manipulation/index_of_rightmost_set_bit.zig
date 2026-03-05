//! Index of Rightmost Set Bit - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/index_of_rightmost_set_bit.py

const std = @import("std");
const testing = std.testing;

pub const RightmostBitError = error{NegativeInput};

/// Returns zero-based index of the rightmost set bit.
/// Returns -1 when number is zero.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn indexOfRightmostSetBit(number: i64) RightmostBitError!i32 {
    if (number < 0) return RightmostBitError.NegativeInput;
    if (number == 0) return -1;

    const unsigned_number: u64 = @intCast(number);
    var intermediate = unsigned_number & ~(@as(u64, unsigned_number - 1));

    var index: i32 = 0;
    while (intermediate != 0) {
        intermediate >>= 1;
        index += 1;
    }

    return index - 1;
}

test "rightmost set bit index: python examples" {
    try testing.expectEqual(@as(i32, -1), try indexOfRightmostSetBit(0));
    try testing.expectEqual(@as(i32, 0), try indexOfRightmostSetBit(5));
    try testing.expectEqual(@as(i32, 2), try indexOfRightmostSetBit(36));
    try testing.expectEqual(@as(i32, 3), try indexOfRightmostSetBit(8));
}

test "rightmost set bit index: invalid negative" {
    try testing.expectError(RightmostBitError.NegativeInput, indexOfRightmostSetBit(-18));
}

test "rightmost set bit index: extreme values" {
    try testing.expectEqual(@as(i32, 0), try indexOfRightmostSetBit(std.math.maxInt(i64)));
    try testing.expectEqual(@as(i32, 62), try indexOfRightmostSetBit(@as(i64, 1) << 62));
}
