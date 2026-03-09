//! Double Linear Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/double_linear_search.py

const std = @import("std");
const testing = std.testing;

/// Searches from both ends and returns the first matching index found by the
/// Python-reference scan order.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn doubleLinearSearch(comptime T: type, items: []const T, target: T) ?usize {
    var left: usize = 0;
    var right: usize = if (items.len == 0) 0 else items.len - 1;

    while (items.len != 0 and left <= right) {
        if (items[left] == target) return left;
        if (items[right] == target) return right;
        left += 1;
        if (right == 0) break;
        right -= 1;
    }

    return null;
}

test "double linear search: examples" {
    const arr = [_]i32{ 1, 5, 5, 10 };
    try testing.expectEqual(@as(?usize, 0), doubleLinearSearch(i32, &arr, 1));
    try testing.expectEqual(@as(?usize, 1), doubleLinearSearch(i32, &arr, 5));
    try testing.expectEqual(@as(?usize, 3), doubleLinearSearch(i32, &arr, 10));
    try testing.expectEqual(@as(?usize, null), doubleLinearSearch(i32, &arr, 100));
}

test "double linear search: boundaries" {
    const empty = [_]i32{};
    try testing.expectEqual(@as(?usize, null), doubleLinearSearch(i32, &empty, 1));

    const single = [_]i32{42};
    try testing.expectEqual(@as(?usize, 0), doubleLinearSearch(i32, &single, 42));
    try testing.expectEqual(@as(?usize, null), doubleLinearSearch(i32, &single, 7));
}
