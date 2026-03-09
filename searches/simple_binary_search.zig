//! Simple Binary Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/simple_binary_search.py

const std = @import("std");
const testing = std.testing;
const existing = @import("binary_search.zig");

/// Returns true if target exists in a sorted slice.
///
/// Time complexity: O(log n)
/// Space complexity: O(log n) in Python reference, O(1) in this Zig version
pub fn binarySearch(comptime T: type, items: []const T, target: T) bool {
    return existing.binarySearch(T, items, target) != null;
}

test "simple binary search: examples" {
    const arr = [_]i32{ 0, 1, 2, 8, 13, 17, 19, 32, 42 };
    try testing.expect(!binarySearch(i32, &arr, 3));
    try testing.expect(binarySearch(i32, &arr, 13));
    try testing.expect(binarySearch(i32, &[_]i32{ 4, 4, 5, 6, 7 }, 4));
    try testing.expect(!binarySearch(i32, &[_]i32{ 4, 4, 5, 6, 7 }, -10));
}

test "simple binary search: boundaries" {
    try testing.expect(binarySearch(i32, &[_]i32{ -18, 2 }, -18));
    try testing.expect(binarySearch(i32, &[_]i32{5}, 5));
    try testing.expect(!binarySearch(i32, &[_]i32{}, 1));
    try testing.expect(binarySearch(f64, &[_]f64{ -0.1, 0.1, 0.8 }, 0.1));
}
