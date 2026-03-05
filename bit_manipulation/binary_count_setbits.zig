//! Binary Count Set Bits - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_count_setbits.py

const std = @import("std");
const testing = std.testing;

pub const BinaryCountSetBitsError = error{NegativeValue};

/// Counts number of set bits in a non-negative integer.
///
/// Time complexity: O(w), w = bit width
/// Space complexity: O(1)
pub fn binaryCountSetBits(value: i64) BinaryCountSetBitsError!u32 {
    if (value < 0) return BinaryCountSetBitsError.NegativeValue;
    return @popCount(@as(u64, @intCast(value)));
}

test "binary count setbits: python examples" {
    try testing.expectEqual(@as(u32, 3), try binaryCountSetBits(25));
    try testing.expectEqual(@as(u32, 2), try binaryCountSetBits(36));
    try testing.expectEqual(@as(u32, 1), try binaryCountSetBits(16));
    try testing.expectEqual(@as(u32, 4), try binaryCountSetBits(58));
    try testing.expectEqual(@as(u32, 32), try binaryCountSetBits(4294967295));
    try testing.expectEqual(@as(u32, 0), try binaryCountSetBits(0));
}

test "binary count setbits: invalid and extreme" {
    try testing.expectError(BinaryCountSetBitsError.NegativeValue, binaryCountSetBits(-10));
    try testing.expectEqual(@as(u32, 63), try binaryCountSetBits(std.math.maxInt(i64)));
}
