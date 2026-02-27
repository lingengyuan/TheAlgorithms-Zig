//! Gnome Sort (Stupid Sort) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/gnome_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place gnome sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn gnomeSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;
    var i: usize = 1;
    while (i < arr.len) {
        if (arr[i - 1] <= arr[i]) {
            i += 1;
        } else {
            const tmp = arr[i];
            arr[i] = arr[i - 1];
            arr[i - 1] = tmp;
            if (i > 1) {
                i -= 1;
            }
        }
    }
}

test "gnome sort: basic case" {
    var arr = [_]i32{ 0, 5, 3, 2, 2 };
    gnomeSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &arr);
}

test "gnome sort: empty array" {
    var arr = [_]i32{};
    gnomeSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "gnome sort: all negative" {
    var arr = [_]i32{ -2, -5, -45 };
    gnomeSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &arr);
}

test "gnome sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    gnomeSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "gnome sort: single element" {
    var arr = [_]i32{42};
    gnomeSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
