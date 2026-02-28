//! Find Unique Number (XOR trick) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/find_unique_number.py

const std = @import("std");
const testing = std.testing;

/// Returns the element that appears exactly once; all others appear exactly twice.
/// XOR of all elements cancels duplicate pairs, leaving the unique number.
/// Time complexity: O(n), Space complexity: O(1)
pub fn findUniqueNumber(arr: []const i32) ?i32 {
    if (arr.len == 0) return null;
    var result: i32 = 0;
    for (arr) |v| result ^= v;
    return result;
}

test "find unique: basic" {
    try testing.expectEqual(@as(?i32, 3), findUniqueNumber(&[_]i32{ 1, 1, 2, 2, 3 }));
    try testing.expectEqual(@as(?i32, 5), findUniqueNumber(&[_]i32{ 4, 5, 4, 6, 6 }));
    try testing.expectEqual(@as(?i32, 7), findUniqueNumber(&[_]i32{7}));
    try testing.expectEqual(@as(?i32, 20), findUniqueNumber(&[_]i32{ 10, 20, 10 }));
}

test "find unique: empty" {
    try testing.expectEqual(@as(?i32, null), findUniqueNumber(&[_]i32{}));
}

test "find unique: negative numbers" {
    try testing.expectEqual(@as(?i32, -3), findUniqueNumber(&[_]i32{ -1, -2, -1, -2, -3 }));
}
