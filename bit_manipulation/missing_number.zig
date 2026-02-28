//! Missing Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/missing_number.py
//! Given a slice containing n distinct integers in range [0, n],
//! finds the one missing number using XOR.

const std = @import("std");
const testing = std.testing;

/// Finds the missing number in [0..n] from a slice of n distinct integers.
/// XOR all indices 0..n with all values; unpaired index is the answer.
/// Time complexity: O(n), Space complexity: O(1)
pub fn missingNumber(nums: []const usize) usize {
    const n = nums.len;
    var result: usize = n; // start with n (the "extra" expected value)
    for (nums, 0..) |v, i| {
        result ^= i ^ v;
    }
    return result;
}

test "missing number: basic" {
    try testing.expectEqual(@as(usize, 2), missingNumber(&[_]usize{ 0, 1, 3 }));
    try testing.expectEqual(@as(usize, 2), missingNumber(&[_]usize{ 3, 0, 1 }));
    try testing.expectEqual(@as(usize, 8), missingNumber(&[_]usize{ 9, 6, 4, 2, 3, 5, 7, 0, 1 }));
}

test "missing number: missing zero" {
    try testing.expectEqual(@as(usize, 0), missingNumber(&[_]usize{ 1, 2, 3 }));
}

test "missing number: missing last" {
    try testing.expectEqual(@as(usize, 3), missingNumber(&[_]usize{ 0, 1, 2 }));
}

test "missing number: single element" {
    try testing.expectEqual(@as(usize, 0), missingNumber(&[_]usize{1}));
    try testing.expectEqual(@as(usize, 1), missingNumber(&[_]usize{0}));
}
