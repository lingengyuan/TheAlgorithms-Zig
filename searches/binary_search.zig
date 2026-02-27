//! Binary Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/simple_binary_search.py

const std = @import("std");
const testing = std.testing;

/// Binary search on a sorted slice. Returns the index of target, or null if not found.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn binarySearch(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;

    var low: usize = 0;
    var high: usize = items.len - 1;

    while (low <= high) {
        const mid = low + (high - low) / 2;
        if (items[mid] == target) {
            return mid;
        } else if (items[mid] < target) {
            low = mid + 1;
        } else {
            // items[mid] > target
            if (mid == 0) break;
            high = mid - 1;
        }
    }
    return null;
}

// ===== Tests =====

test "binary search: found in middle" {
    const arr = [_]i32{ 0, 1, 2, 8, 13, 17, 19, 32, 42 };
    try testing.expectEqual(@as(?usize, 4), binarySearch(i32, &arr, 13));
}

test "binary search: not found" {
    const arr = [_]i32{ 0, 1, 2, 8, 13, 17, 19, 32, 42 };
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 3));
}

test "binary search: found duplicate" {
    const arr = [_]i32{ 4, 4, 5, 6, 7 };
    const result = binarySearch(i32, &arr, 4);
    try testing.expect(result != null);
    try testing.expectEqual(@as(i32, 4), arr[result.?]);
}

test "binary search: not found negative" {
    const arr = [_]i32{ 4, 4, 5, 6, 7 };
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, -10));
}

test "binary search: found at start" {
    const arr = [_]i32{ -18, 2 };
    try testing.expectEqual(@as(?usize, 0), binarySearch(i32, &arr, -18));
}

test "binary search: single element found" {
    const arr = [_]i32{5};
    try testing.expectEqual(@as(?usize, 0), binarySearch(i32, &arr, 5));
}

test "binary search: single element not found" {
    const arr = [_]i32{5};
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 3));
}

test "binary search: empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), binarySearch(i32, &arr, 1));
}

test "binary search: found at end" {
    const arr = [_]i32{ 1, 3, 5, 7, 9 };
    try testing.expectEqual(@as(?usize, 4), binarySearch(i32, &arr, 9));
}
