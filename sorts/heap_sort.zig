//! Heap Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/heap_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place heap sort (max-heap), ascending order.
/// Time complexity: O(n log n), Space complexity: O(1)
pub fn heapSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var start = arr.len / 2;
    while (start > 0) {
        start -= 1;
        siftDown(T, arr, start, arr.len);
    }

    var end = arr.len;
    while (end > 1) {
        end -= 1;
        std.mem.swap(T, &arr[0], &arr[end]);
        siftDown(T, arr, 0, end);
    }
}

fn siftDown(comptime T: type, arr: []T, start: usize, end: usize) void {
    var root = start;

    while (true) {
        const left = root * 2 + 1;
        if (left >= end) return;

        var child = left;
        const right = left + 1;
        if (right < end and arr[right] > arr[left]) {
            child = right;
        }

        if (arr[root] < arr[child]) {
            std.mem.swap(T, &arr[root], &arr[child]);
            root = child;
        } else {
            return;
        }
    }
}

test "heap sort: basic case" {
    var arr = [_]i32{ 0, 5, 3, 2, 2 };
    heapSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "heap sort: empty array" {
    var arr = [_]i32{};
    heapSort(i32, &arr);
    try testing.expectEqual(@as(usize, 0), arr.len);
}

test "heap sort: all negative" {
    var arr = [_]i32{ -2, -5, -45 };
    heapSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "heap sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    heapSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "heap sort: reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    heapSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "heap sort: single element" {
    var arr = [_]i32{42};
    heapSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
