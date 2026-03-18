//! Merge Sort (Divide and Conquer) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/mergesort.py

const std = @import("std");
const testing = std.testing;

/// Merges two sorted slices into a new sorted slice.
///
/// Time complexity: O(n + m)
/// Space complexity: O(n + m)
pub fn merge(
    allocator: std.mem.Allocator,
    left: []const i64,
    right: []const i64,
) std.mem.Allocator.Error![]i64 {
    const out = try allocator.alloc(i64, left.len + right.len);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < left.len and j < right.len) {
        if (left[i] < right[j]) {
            out[k] = left[i];
            i += 1;
        } else {
            out[k] = right[j];
            j += 1;
        }
        k += 1;
    }

    while (i < left.len) : (i += 1) {
        out[k] = left[i];
        k += 1;
    }
    while (j < right.len) : (j += 1) {
        out[k] = right[j];
        k += 1;
    }

    return out;
}

/// Returns a sorted copy of `array`.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n)
pub fn mergeSort(allocator: std.mem.Allocator, array: []const i64) std.mem.Allocator.Error![]i64 {
    if (array.len <= 1) return allocator.dupe(i64, array);

    const middle = array.len / 2;
    const left_half = try mergeSort(allocator, array[0..middle]);
    defer allocator.free(left_half);

    const right_half = try mergeSort(allocator, array[middle..]);
    defer allocator.free(right_half);

    return merge(allocator, left_half, right_half);
}

test "mergesort: merge helper" {
    const alloc = testing.allocator;

    const m1 = try merge(alloc, &[_]i64{-2}, &[_]i64{-1});
    defer alloc.free(m1);
    try testing.expectEqualSlices(i64, &[_]i64{ -2, -1 }, m1);

    const m2 = try merge(alloc, &[_]i64{ 12, 15 }, &[_]i64{ 13, 14 });
    defer alloc.free(m2);
    try testing.expectEqualSlices(i64, &[_]i64{ 12, 13, 14, 15 }, m2);

    const m3 = try merge(alloc, &[_]i64{}, &[_]i64{});
    defer alloc.free(m3);
    try testing.expectEqual(@as(usize, 0), m3.len);
}

test "mergesort: python-style examples" {
    const alloc = testing.allocator;

    const s1 = try mergeSort(alloc, &[_]i64{ -2, 3, -10, 11, 99, 100000, 100, -200 });
    defer alloc.free(s1);
    try testing.expectEqualSlices(i64, &[_]i64{ -200, -10, -2, 3, 11, 99, 100, 100000 }, s1);

    const s2 = try mergeSort(alloc, &[_]i64{-200});
    defer alloc.free(s2);
    try testing.expectEqualSlices(i64, &[_]i64{-200}, s2);

    const s3 = try mergeSort(alloc, &[_]i64{});
    defer alloc.free(s3);
    try testing.expectEqual(@as(usize, 0), s3.len);
}

test "mergesort: extreme already descending" {
    const alloc = testing.allocator;

    var arr: [1024]i64 = undefined;
    for (0..arr.len) |i| arr[i] = @as(i64, @intCast(arr.len - i));

    const sorted = try mergeSort(alloc, arr[0..]);
    defer alloc.free(sorted);

    for (0..sorted.len - 1) |i| {
        try testing.expect(sorted[i] <= sorted[i + 1]);
    }
}
