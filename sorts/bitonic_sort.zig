//! Bitonic Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/bitonic_sort.py

const std = @import("std");
const testing = std.testing;

pub const Direction = enum(u1) {
    descending = 0,
    ascending = 1,
};

fn compAndSwap(comptime T: type, arr: []T, i: usize, j: usize, direction: Direction) void {
    const should_swap = (direction == .ascending and arr[i] > arr[j]) or
        (direction == .descending and arr[i] < arr[j]);
    if (should_swap) std.mem.swap(T, &arr[i], &arr[j]);
}

/// Merges a bitonic sequence in the given direction.
/// Time complexity: O(n log n), Space complexity: O(log n) recursion
pub fn bitonicMerge(comptime T: type, arr: []T, low: usize, length: usize, direction: Direction) void {
    if (length <= 1) return;

    const middle = length / 2;
    for (low..low + middle) |i| {
        compAndSwap(T, arr, i, i + middle, direction);
    }
    bitonicMerge(T, arr, low, middle, direction);
    bitonicMerge(T, arr, low + middle, middle, direction);
}

/// Bitonic sort for ranges where length is power of 2.
/// Time complexity: O(n log² n), Space complexity: O(log n) recursion
pub fn bitonicSortRange(comptime T: type, arr: []T, low: usize, length: usize, direction: Direction) void {
    if (length <= 1) return;

    const middle = length / 2;
    bitonicSortRange(T, arr, low, middle, .ascending);
    bitonicSortRange(T, arr, low + middle, middle, .descending);
    bitonicMerge(T, arr, low, length, direction);
}

/// Convenience full-array ascending sort.
pub fn bitonicSort(comptime T: type, arr: []T) void {
    bitonicSortRange(T, arr, 0, arr.len, .ascending);
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}

test "bitonic sort: python reference examples" {
    var arr1 = [_]i32{ 12, 34, 92, -23, 0, -121, -167, 145 };
    bitonicSortRange(i32, &arr1, 0, arr1.len, .ascending);
    try testing.expectEqualSlices(i32, &[_]i32{ -167, -121, -23, 0, 12, 34, 92, 145 }, &arr1);

    bitonicSortRange(i32, &arr1, 0, arr1.len, .descending);
    try testing.expectEqualSlices(i32, &[_]i32{ 145, 92, 34, 12, 0, -23, -121, -167 }, &arr1);
}

test "bitonic sort: comp and merge examples" {
    var arr = [_]i32{ 12, 42, -21, 1 };
    compAndSwap(i32, &arr, 1, 2, .ascending);
    try testing.expectEqualSlices(i32, &[_]i32{ 12, -21, 42, 1 }, &arr);

    compAndSwap(i32, &arr, 1, 2, .descending);
    try testing.expectEqualSlices(i32, &[_]i32{ 12, 42, -21, 1 }, &arr);

    compAndSwap(i32, &arr, 0, 3, .ascending);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 42, -21, 12 }, &arr);

    compAndSwap(i32, &arr, 0, 3, .descending);
    try testing.expectEqualSlices(i32, &[_]i32{ 12, 42, -21, 1 }, &arr);

    bitonicMerge(i32, &arr, 0, 4, .ascending);
    try testing.expectEqualSlices(i32, &[_]i32{ -21, 1, 12, 42 }, &arr);
}

test "bitonic sort: extreme power-of-two input" {
    const alloc = testing.allocator;
    const n: usize = 16_384;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| v.* = @as(i32, @intCast(n - i));

    bitonicSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
