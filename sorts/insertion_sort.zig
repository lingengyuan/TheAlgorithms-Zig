//! Insertion Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/insertion_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place insertion sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn insertionSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        const key = arr[i];
        var j: usize = i;
        while (j > 0 and arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

// ===== Tests =====

test "insertion sort: basic case" {
    var arr = [_]i32{ 0, 5, 3, 2, 2 };
    insertionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "insertion sort: empty array" {
    var arr = [_]i32{};
    insertionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "insertion sort: all negative" {
    var arr = [_]i32{ -2, -5, -45 };
    insertionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "insertion sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    insertionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "insertion sort: reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    insertionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "insertion sort: single element" {
    var arr = [_]i32{42};
    insertionSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
