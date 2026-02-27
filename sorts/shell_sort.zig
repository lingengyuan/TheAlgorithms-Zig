//! Shell Sort - Zig implementation (Ciura gap sequence)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/shell_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place shell sort using Marcin Ciura's gap sequence, ascending order.
/// Time complexity: O(n^1.3) average, Space complexity: O(1)
pub fn shellSort(comptime T: type, arr: []T) void {
    const gaps = [_]usize{ 701, 301, 132, 57, 23, 10, 4, 1 };
    for (gaps) |gap| {
        if (gap >= arr.len) continue;
        var i: usize = gap;
        while (i < arr.len) : (i += 1) {
            const insert_value = arr[i];
            var j = i;
            while (j >= gap and arr[j - gap] > insert_value) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = insert_value;
        }
    }
}

test "shell sort: basic case" {
    var arr = [_]i32{ 0, 5, 3, 2, 2 };
    shellSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "shell sort: empty array" {
    var arr = [_]i32{};
    shellSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "shell sort: all negative" {
    var arr = [_]i32{ -2, -5, -45 };
    shellSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "shell sort: reverse sorted" {
    var arr = [_]i32{ 9, 7, 5, 3, 1 };
    shellSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 5, 7, 9 }, &arr);
}

test "shell sort: single element" {
    var arr = [_]i32{42};
    shellSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "shell sort: large array" {
    var arr = [_]i32{ 38, 27, 43, 3, 9, 82, 10, 1, 57, 24 };
    shellSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 9, 10, 24, 27, 38, 43, 57, 82 }, &arr);
}
