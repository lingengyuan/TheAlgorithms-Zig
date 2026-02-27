//! Ternary Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/ternary_search.py

const std = @import("std");
const testing = std.testing;

/// Iterative ternary search on a sorted slice. Returns the index of target, or null.
/// Time complexity: O(log₃ n), Space complexity: O(1)
pub fn ternarySearch(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;

    var left: usize = 0;
    var right: usize = items.len - 1;

    while (left <= right) {
        if (right - left < 3) {
            // Linear scan for small ranges
            var i = left;
            while (i <= right) : (i += 1) {
                if (items[i] == target) return i;
            }
            return null;
        }
        const third = (right - left) / 3;
        const mid1 = left + third;
        const mid2 = right - third;

        if (items[mid1] == target) return mid1;
        if (items[mid2] == target) return mid2;

        if (target < items[mid1]) {
            right = mid1 - 1;
        } else if (target > items[mid2]) {
            left = mid2 + 1;
        } else {
            left = mid1 + 1;
            right = mid2 - 1;
        }
    }
    return null;
}

test "ternary search: found in middle" {
    const arr = [_]i32{ 0, 1, 2, 8, 13, 17, 19, 32, 42 };
    try testing.expectEqual(@as(?usize, 4), ternarySearch(i32, &arr, 13));
}

test "ternary search: not found" {
    const arr = [_]i32{ 0, 1, 2, 8, 13, 17, 19, 32, 42 };
    try testing.expectEqual(@as(?usize, null), ternarySearch(i32, &arr, 3));
}

test "ternary search: found at start" {
    const arr = [_]i32{ 4, 5, 6, 7 };
    try testing.expectEqual(@as(?usize, 0), ternarySearch(i32, &arr, 4));
}

test "ternary search: found negative" {
    const arr = [_]i32{ -18, 2 };
    try testing.expectEqual(@as(?usize, 0), ternarySearch(i32, &arr, -18));
}

test "ternary search: single element" {
    const arr = [_]i32{5};
    try testing.expectEqual(@as(?usize, 0), ternarySearch(i32, &arr, 5));
}

test "ternary search: empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), ternarySearch(i32, &arr, 1));
}

test "ternary search: not found in range" {
    const arr = [_]i32{ 4, 4, 5, 6, 7 };
    try testing.expectEqual(@as(?usize, null), ternarySearch(i32, &arr, -10));
}
