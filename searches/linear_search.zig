//! Linear Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/linear_search.py

const std = @import("std");
const testing = std.testing;

/// Linear search: returns the index of target in the slice, or null if not found.
/// Time complexity: O(n), Space complexity: O(1)
pub fn linearSearch(comptime T: type, items: []const T, target: T) ?usize {
    for (items, 0..) |item, i| {
        if (item == target) return i;
    }
    return null;
}

// ===== Tests =====

test "linear search: found at beginning" {
    const arr = [_]i32{ 0, 5, 7, 10, 15 };
    try testing.expectEqual(@as(?usize, 0), linearSearch(i32, &arr, 0));
}

test "linear search: found at end" {
    const arr = [_]i32{ 0, 5, 7, 10, 15 };
    try testing.expectEqual(@as(?usize, 4), linearSearch(i32, &arr, 15));
}

test "linear search: found in middle" {
    const arr = [_]i32{ 0, 5, 7, 10, 15 };
    try testing.expectEqual(@as(?usize, 1), linearSearch(i32, &arr, 5));
}

test "linear search: not found" {
    const arr = [_]i32{ 0, 5, 7, 10, 15 };
    try testing.expectEqual(@as(?usize, null), linearSearch(i32, &arr, 6));
}

test "linear search: empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), linearSearch(i32, &arr, 1));
}

test "linear search: single element found" {
    const arr = [_]i32{42};
    try testing.expectEqual(@as(?usize, 0), linearSearch(i32, &arr, 42));
}

test "linear search: single element not found" {
    const arr = [_]i32{42};
    try testing.expectEqual(@as(?usize, null), linearSearch(i32, &arr, 99));
}
