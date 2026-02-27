//! Quick Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place quick sort using Lomuto partition, ascending order.
/// Time complexity: O(n log n) average, O(n²) worst, Space complexity: O(log n) average (recursion)
pub fn quickSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    const pivot_idx = partition(T, arr);
    quickSort(T, arr[0..pivot_idx]);
    if (pivot_idx + 1 < arr.len) {
        quickSort(T, arr[(pivot_idx + 1)..]);
    }
}

fn partition(comptime T: type, arr: []T) usize {
    const pivot = arr[arr.len - 1];
    var i: usize = 0;

    for (0..(arr.len - 1)) |j| {
        if (arr[j] <= pivot) {
            std.mem.swap(T, &arr[i], &arr[j]);
            i += 1;
        }
    }

    std.mem.swap(T, &arr[i], &arr[arr.len - 1]);
    return i;
}

test "quick sort: basic case" {
    var arr = [_]i32{ 0, 5, 3, 2, 2 };
    quickSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "quick sort: empty array" {
    var arr = [_]i32{};
    quickSort(i32, &arr);
    try testing.expectEqual(@as(usize, 0), arr.len);
}

test "quick sort: all negative" {
    var arr = [_]i32{ -2, -5, -45 };
    quickSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "quick sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    quickSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "quick sort: reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    quickSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "quick sort: single element" {
    var arr = [_]i32{42};
    quickSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
