//! Stooge Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/stooge_sort.py

const std = @import("std");
const testing = std.testing;

fn stooge(comptime T: type, arr: []T, i: usize, h: usize) void {
    if (i >= h) return;

    if (arr[i] > arr[h]) {
        std.mem.swap(T, &arr[i], &arr[h]);
    }

    if (h - i + 1 > 2) {
        const t = (h - i + 1) / 3;
        stooge(T, arr, i, h - t);
        stooge(T, arr, i + t, h);
        stooge(T, arr, i, h - t);
    }
}

/// In-place stooge sort, ascending order.
/// Time complexity: O(n^(log 3 / log 1.5)) ~ O(n^2.7095), Space complexity: O(log n) recursion
pub fn stoogeSort(comptime T: type, arr: []T) void {
    if (arr.len == 0) return;
    stooge(T, arr, 0, arr.len - 1);
}

test "stooge sort: python reference examples" {
    var a1 = [_]f64{ 18.1, 0.0, -7.1, -1.0, 2.0, 2.0 };
    stoogeSort(f64, &a1);
    try testing.expectEqualSlices(f64, &[_]f64{ -7.1, -1.0, 0.0, 2.0, 2.0, 18.1 }, &a1);

    var a2 = [_]i32{};
    stoogeSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);
}

test "stooge sort: edge cases" {
    var one = [_]i32{42};
    stoogeSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &one);

    var dup = [_]i32{ 3, 3, 3, 3 };
    stoogeSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 3, 3, 3 }, &dup);
}

test "stooge sort: extreme small-size reverse input" {
    var arr = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    stoogeSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &arr);
}
