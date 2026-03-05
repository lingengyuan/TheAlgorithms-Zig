//! Binary Count Trailing Zeros - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_count_trailing_zeros.py

const std = @import("std");
const testing = std.testing;

pub const BinaryTrailingZerosError = error{NegativeValue};

/// Returns the number of trailing zero bits in the binary representation.
/// Returns `0` for input `0`.
///
/// Time complexity: O(k), where k is the number of trailing zeros.
/// Space complexity: O(1)
pub fn binaryCountTrailingZeros(value: i64) BinaryTrailingZerosError!u8 {
    if (value < 0) return BinaryTrailingZerosError.NegativeValue;
    if (value == 0) return 0;

    var n: u64 = @intCast(value);
    var count: u8 = 0;
    while ((n & 1) == 0) : (count += 1) {
        n >>= 1;
    }
    return count;
}

test "binary count trailing zeros: python examples" {
    try testing.expectEqual(@as(u8, 0), try binaryCountTrailingZeros(25));
    try testing.expectEqual(@as(u8, 2), try binaryCountTrailingZeros(36));
    try testing.expectEqual(@as(u8, 4), try binaryCountTrailingZeros(16));
    try testing.expectEqual(@as(u8, 1), try binaryCountTrailingZeros(58));
    try testing.expectEqual(@as(u8, 32), try binaryCountTrailingZeros(4_294_967_296));
    try testing.expectEqual(@as(u8, 0), try binaryCountTrailingZeros(0));
}

test "binary count trailing zeros: invalid and extreme" {
    try testing.expectError(BinaryTrailingZerosError.NegativeValue, binaryCountTrailingZeros(-10));

    try testing.expectEqual(@as(u8, 62), try binaryCountTrailingZeros(@as(i64, 1) << 62));
    try testing.expectEqual(@as(u8, 0), try binaryCountTrailingZeros(std.math.maxInt(i64)));
}
