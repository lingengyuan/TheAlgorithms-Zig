//! Highest Set Bit Position - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/highest_set_bit.py

const std = @import("std");
const testing = std.testing;

pub const HighestSetBitError = error{NegativeInput};

/// Returns 1-based position of the highest set bit.
/// Returns 0 for input 0.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn highestSetBitPosition(number: i64) HighestSetBitError!u8 {
    if (number < 0) return HighestSetBitError.NegativeInput;

    var value: u64 = @intCast(number);
    var position: u8 = 0;

    while (value != 0) {
        position += 1;
        value >>= 1;
    }

    return position;
}

test "highest set bit: known values" {
    try testing.expectEqual(@as(u8, 5), try highestSetBitPosition(25));
    try testing.expectEqual(@as(u8, 6), try highestSetBitPosition(37));
    try testing.expectEqual(@as(u8, 1), try highestSetBitPosition(1));
    try testing.expectEqual(@as(u8, 3), try highestSetBitPosition(4));
    try testing.expectEqual(@as(u8, 0), try highestSetBitPosition(0));
}

test "highest set bit: invalid negative" {
    try testing.expectError(HighestSetBitError.NegativeInput, highestSetBitPosition(-1));
}

test "highest set bit: extreme values" {
    try testing.expectEqual(@as(u8, 63), try highestSetBitPosition(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u8, 62), try highestSetBitPosition((@as(i64, 1) << 61) + 7));
}
