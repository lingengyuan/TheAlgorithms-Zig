//! Selection Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/selection_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place selection sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn selectionSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;
    for (0..arr.len - 1) |i| {
        var min_idx = i;
        for (i + 1..arr.len) |k| {
            if (arr[k] < arr[min_idx]) {
                min_idx = k;
            }
        }
        if (min_idx != i) {
            const tmp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = tmp;
        }
    }
}

test "selection sort: basic case" {
    var arr = [_]i32{ 0, 5, 3, 2, 2 };
    selectionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "selection sort: empty array" {
    var arr = [_]i32{};
    selectionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "selection sort: all negative" {
    var arr = [_]i32{ -2, -5, -45 };
    selectionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "selection sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    selectionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "selection sort: single element" {
    var arr = [_]i32{42};
    selectionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
