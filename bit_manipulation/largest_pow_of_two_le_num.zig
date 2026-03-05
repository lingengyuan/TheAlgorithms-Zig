//! Largest Power of Two <= Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/largest_pow_of_two_le_num.py

const std = @import("std");
const testing = std.testing;

/// Returns the largest power of two less than or equal to `number`.
/// For non-positive input, returns 0.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn largestPowOfTwoLeNum(number: i64) u64 {
    if (number <= 0) return 0;

    var result: u64 = 1;
    const target: u64 = @intCast(number);
    while ((result << 1) <= target) {
        result <<= 1;
    }
    return result;
}

test "largest power of two <= num: python examples" {
    try testing.expectEqual(@as(u64, 0), largestPowOfTwoLeNum(0));
    try testing.expectEqual(@as(u64, 1), largestPowOfTwoLeNum(1));
    try testing.expectEqual(@as(u64, 0), largestPowOfTwoLeNum(-1));
    try testing.expectEqual(@as(u64, 2), largestPowOfTwoLeNum(3));
    try testing.expectEqual(@as(u64, 8), largestPowOfTwoLeNum(15));
    try testing.expectEqual(@as(u64, 64), largestPowOfTwoLeNum(99));
    try testing.expectEqual(@as(u64, 128), largestPowOfTwoLeNum(178));
    try testing.expectEqual(@as(u64, 524288), largestPowOfTwoLeNum(999999));
}

test "largest power of two <= num: boundary and extreme" {
    try testing.expectEqual(@as(u64, 1) << 62, largestPowOfTwoLeNum(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u64, 0), largestPowOfTwoLeNum(std.math.minInt(i64)));
}
