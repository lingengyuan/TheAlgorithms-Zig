//! Recursive Insertion Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/recursive_insertion_sort.py

const std = @import("std");
const testing = std.testing;

/// Inserts element at `index - 1` into sorted suffix by adjacent swaps.
fn insertNext(comptime T: type, arr: []T, index: usize) void {
    if (arr.len == 0) return;
    if (index >= arr.len) return;
    if (index == 0) return;
    if (arr[index - 1] <= arr[index]) return;

    std.mem.swap(T, &arr[index - 1], &arr[index]);
    insertNext(T, arr, index + 1);
}

/// Recursive insertion sort with explicit `n` parameter.
/// Time complexity: O(n²), Space complexity: O(n) recursion depth
pub fn recInsertionSort(comptime T: type, arr: []T, n: usize) void {
    if (arr.len <= 1 or n <= 1) return;

    insertNext(T, arr, n - 1);
    recInsertionSort(T, arr, n - 1);
}

/// Convenience wrapper for full slice sort.
pub fn recursiveInsertionSort(comptime T: type, arr: []T) void {
    recInsertionSort(T, arr, arr.len);
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "recursive insertion sort: python reference examples" {
    var col1 = [_]i32{ 1, 2, 1 };
    recInsertionSort(i32, &col1, col1.len);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2 }, &col1);

    var col2 = [_]i32{ 2, 1, 0, -1, -2 };
    recInsertionSort(i32, &col2, col2.len);
    try testing.expectEqualSlices(i32, &[_]i32{ -2, -1, 0, 1, 2 }, &col2);

    var col3 = [_]i32{1};
    recInsertionSort(i32, &col3, col3.len);
    try testing.expectEqualSlices(i32, &[_]i32{1}, &col3);
}

test "recursive insertion sort: insert next examples" {
    var col1 = [_]i32{ 3, 2, 4, 2 };
    insertNext(i32, &col1, 1);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 3, 4, 2 }, &col1);

    var col2 = [_]i32{ 3, 2, 3 };
    insertNext(i32, &col2, 2);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 2, 3 }, &col2);

    var col3 = [_]i32{};
    insertNext(i32, &col3, 1);
    try testing.expectEqual(@as(usize, 0), col3.len);
}

test "recursive insertion sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 1500;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    recursiveInsertionSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
