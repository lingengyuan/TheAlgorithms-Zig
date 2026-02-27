//! Exponential Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/exponential_search.py

const std = @import("std");
const testing = std.testing;

/// Exponential search on a sorted slice. Returns the index of target, or null if not found.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn exponentialSearch(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;
    if (items[0] == target) return 0;

    var bound: usize = 1;
    while (bound < items.len and items[bound] < target) {
        bound *= 2;
    }

    const left = bound / 2;
    const right = @min(bound, items.len - 1);
    return binarySearchRange(T, items, target, left, right);
}

fn binarySearchRange(comptime T: type, items: []const T, target: T, left: usize, right: usize) ?usize {
    var low = left;
    var high = right;

    while (low <= high) {
        const mid = low + (high - low) / 2;
        if (items[mid] == target) {
            return mid;
        } else if (items[mid] < target) {
            low = mid + 1;
        } else {
            if (mid == 0) break;
            high = mid - 1;
        }
    }
    return null;
}

test "exponential search: found in middle" {
    const arr = [_]i32{ 2, 3, 4, 10, 40, 50, 60, 70 };
    try testing.expectEqual(@as(?usize, 4), exponentialSearch(i32, &arr, 40));
}

test "exponential search: found at beginning" {
    const arr = [_]i32{ 2, 3, 4, 10, 40 };
    try testing.expectEqual(@as(?usize, 0), exponentialSearch(i32, &arr, 2));
}

test "exponential search: found at end" {
    const arr = [_]i32{ 1, 2, 4, 8, 16, 32, 64 };
    try testing.expectEqual(@as(?usize, 6), exponentialSearch(i32, &arr, 64));
}

test "exponential search: not found" {
    const arr = [_]i32{ 2, 3, 4, 10, 40 };
    try testing.expectEqual(@as(?usize, null), exponentialSearch(i32, &arr, 5));
}

test "exponential search: empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), exponentialSearch(i32, &arr, 1));
}

test "exponential search: single element not found" {
    const arr = [_]i32{7};
    try testing.expectEqual(@as(?usize, null), exponentialSearch(i32, &arr, 9));
}
