//! Swap All Odd and Even Bits - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/swap_all_odd_and_even_bits.py

const std = @import("std");
const testing = std.testing;

/// Swaps odd/even bit positions in the low 32-bit representation.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn swapOddEvenBits(num: i64) u32 {
    const low32: u32 = @truncate(@as(u64, @bitCast(num)));
    const even_bits = low32 & 0xAAAAAAAA;
    const odd_bits = low32 & 0x55555555;
    return (even_bits >> 1) | (odd_bits << 1);
}

test "swap odd/even bits: python examples" {
    try testing.expectEqual(@as(u32, 0), swapOddEvenBits(0));
    try testing.expectEqual(@as(u32, 2), swapOddEvenBits(1));
    try testing.expectEqual(@as(u32, 1), swapOddEvenBits(2));
    try testing.expectEqual(@as(u32, 3), swapOddEvenBits(3));
    try testing.expectEqual(@as(u32, 8), swapOddEvenBits(4));
    try testing.expectEqual(@as(u32, 10), swapOddEvenBits(5));
    try testing.expectEqual(@as(u32, 9), swapOddEvenBits(6));
    try testing.expectEqual(@as(u32, 43), swapOddEvenBits(23));
}

test "swap odd/even bits: negative and wide inputs" {
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), swapOddEvenBits(-1));
    try testing.expectEqual(@as(u32, 0xFFFFFFFD), swapOddEvenBits(-2));

    try testing.expectEqual(@as(u32, 0), swapOddEvenBits(@as(i64, 1) << 40));
    try testing.expectEqual(@as(u32, 2), swapOddEvenBits((@as(i64, 1) << 40) + 1));
}
