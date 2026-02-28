//! Count Set Bits (Brian Kernighan's method) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_count_setbits.py

const std = @import("std");
const testing = std.testing;

/// Counts the number of 1-bits in n using Brian Kernighan's algorithm.
/// Each iteration clears the lowest set bit: n &= (n - 1).
/// Time complexity: O(number of set bits)
pub fn countSetBits(n: u64) u64 {
    var count: u64 = 0;
    var x = n;
    while (x != 0) {
        x &= x - 1;
        count += 1;
    }
    return count;
}

test "count set bits: known values" {
    try testing.expectEqual(@as(u64, 3), countSetBits(25));  // 11001
    try testing.expectEqual(@as(u64, 2), countSetBits(36));  // 100100
    try testing.expectEqual(@as(u64, 1), countSetBits(16));  // 10000
    try testing.expectEqual(@as(u64, 4), countSetBits(58));  // 111010
    try testing.expectEqual(@as(u64, 0), countSetBits(0));
    try testing.expectEqual(@as(u64, 1), countSetBits(1));
}

test "count set bits: all ones" {
    try testing.expectEqual(@as(u64, 64), countSetBits(std.math.maxInt(u64)));
}

test "count set bits: powers of two" {
    for (0..16) |i| {
        const n: u64 = @as(u64, 1) << @intCast(i);
        try testing.expectEqual(@as(u64, 1), countSetBits(n));
    }
}
