//! Recursive Merge Sort Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/recursive_mergesort_array.py

const std = @import("std");
const testing = std.testing;

/// Recursively sorts the array in-place using merge sort.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn mergeSortRecursive(comptime T: type, allocator: std.mem.Allocator, arr: []T) !void {
    if (arr.len <= 1) return;

    const middle = arr.len / 2;
    const left = try allocator.alloc(T, middle);
    defer allocator.free(left);
    const right = try allocator.alloc(T, arr.len - middle);
    defer allocator.free(right);

    @memcpy(left, arr[0..middle]);
    @memcpy(right, arr[middle..]);

    try mergeSortRecursive(T, allocator, left);
    try mergeSortRecursive(T, allocator, right);

    var li: usize = 0;
    var ri: usize = 0;
    var i: usize = 0;

    while (li < left.len and ri < right.len) : (i += 1) {
        if (left[li] < right[ri]) {
            arr[i] = left[li];
            li += 1;
        } else {
            arr[i] = right[ri];
            ri += 1;
        }
    }
    while (li < left.len) : (li += 1) {
        arr[i] = left[li];
        i += 1;
    }
    while (ri < right.len) : (ri += 1) {
        arr[i] = right[ri];
        i += 1;
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}

test "recursive mergesort array: python reference examples" {
    const alloc = testing.allocator;

    var a1 = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try mergeSortRecursive(i32, alloc, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &a1);

    var a2 = [_]i32{100};
    try mergeSortRecursive(i32, alloc, &a2);
    try testing.expectEqualSlices(i32, &[_]i32{100}, &a2);

    var a3 = [_]i32{};
    try mergeSortRecursive(i32, alloc, &a3);
    try testing.expectEqual(@as(usize, 0), a3.len);
}

test "recursive mergesort array: edge mixed input" {
    const alloc = testing.allocator;

    var a = [_]i32{ 10, 22, 1, 2, 3, 9, 15, 23 };
    try mergeSortRecursive(i32, alloc, &a);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 9, 10, 15, 22, 23 }, &a);
}

test "recursive mergesort array: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 25_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(123);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i32, -1_000_000, 1_000_000);

    try mergeSortRecursive(i32, alloc, arr);
    try expectSortedAscending(i32, arr);
}
