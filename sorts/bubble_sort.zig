//! Bubble Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/bubble_sort.py

const std = @import("std");
const testing = std.testing;

/// 原地冒泡排序，升序
/// 时间复杂度: O(n²)，空间复杂度: O(1)
pub fn bubbleSort(comptime T: type, arr: []T) void {
    const n = arr.len;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j: usize = 0;
        while (j < n - i - 1) : (j += 1) {
            if (arr[j] > arr[j + 1]) {
                const tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

// ===== Tests =====

test "bubble sort: basic case" {
    var arr = [_]i32{ 0, 5, 2, 3, 2 };
    bubbleSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "bubble sort: empty array" {
    var arr = [_]i32{};
    bubbleSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "bubble sort: all negative" {
    var arr = [_]i32{ -2, -45, -5 };
    bubbleSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "bubble sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    bubbleSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "bubble sort: single element" {
    var arr = [_]i32{42};
    bubbleSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
