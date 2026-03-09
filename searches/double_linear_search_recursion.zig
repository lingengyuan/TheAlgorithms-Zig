//! Double Linear Search (Recursive) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/double_linear_search_recursion.py

const std = @import("std");
const testing = std.testing;

fn searchRange(comptime T: type, items: []const T, key: T, left: usize, right: isize) ?usize {
    if (@as(isize, @intCast(left)) > right) return null;
    if (items[left] == key) return left;
    if (items[@intCast(right)] == key) return @intCast(right);
    return searchRange(T, items, key, left + 1, right - 1);
}

/// Recursive two-sided linear search.
///
/// Time complexity: O(n)
/// Space complexity: O(n) recursion
pub fn search(comptime T: type, items: []const T, key: T) ?usize {
    if (items.len == 0) return null;
    return searchRange(T, items, key, 0, @intCast(items.len - 1));
}

test "double linear recursion: examples" {
    try testing.expectEqual(@as(?usize, 5), search(i32, &[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 5));
    try testing.expectEqual(@as(?usize, 2), search(i32, &[_]i32{ 1, 2, 4, 5, 3 }, 4));
    try testing.expectEqual(@as(?usize, null), search(i32, &[_]i32{ 1, 2, 4, 5, 3 }, 6));
    try testing.expectEqual(@as(?usize, 0), search(i32, &[_]i32{5}, 5));
}

test "double linear recursion: boundaries" {
    try testing.expectEqual(@as(?usize, null), search(i32, &[_]i32{}, 1));
    try testing.expectEqual(@as(?usize, 0), search(i32, &[_]i32{ 7, 8, 9 }, 7));
    try testing.expectEqual(@as(?usize, 2), search(i32, &[_]i32{ 7, 8, 9 }, 9));
}
