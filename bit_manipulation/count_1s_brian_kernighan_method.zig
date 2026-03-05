//! Count 1s (Brian Kernighan Method) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/count_1s_brian_kernighan_method.py

const std = @import("std");
const testing = std.testing;

pub const BrianKernighanError = error{InvalidInput};

/// Counts set bits using Brian Kernighan's algorithm.
///
/// Time complexity: O(k), k = number of set bits
/// Space complexity: O(1)
pub fn get1sCount(number: i64) BrianKernighanError!u32 {
    if (number < 0) return BrianKernighanError.InvalidInput;

    var n: u64 = @intCast(number);
    var count: u32 = 0;
    while (n != 0) {
        n &= n - 1;
        count += 1;
    }
    return count;
}

test "count 1s brian kernighan: python examples" {
    try testing.expectEqual(@as(u32, 3), try get1sCount(25));
    try testing.expectEqual(@as(u32, 3), try get1sCount(37));
    try testing.expectEqual(@as(u32, 3), try get1sCount(21));
    try testing.expectEqual(@as(u32, 4), try get1sCount(58));
    try testing.expectEqual(@as(u32, 0), try get1sCount(0));
    try testing.expectEqual(@as(u32, 1), try get1sCount(256));
}

test "count 1s brian kernighan: invalid and extreme" {
    try testing.expectError(BrianKernighanError.InvalidInput, get1sCount(-1));
    try testing.expectEqual(@as(u32, 63), try get1sCount(std.math.maxInt(i64)));
}
