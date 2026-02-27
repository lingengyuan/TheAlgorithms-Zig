//! Cocktail Shaker Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/cocktail_shaker_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place cocktail shaker sort (bidirectional bubble sort), ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn cocktailShakerSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;
    var start: usize = 0;
    var end: usize = arr.len - 1;

    while (start < end) {
        var swapped = false;

        // Left to right pass
        for (start..end) |i| {
            if (arr[i] > arr[i + 1]) {
                const tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
                swapped = true;
            }
        }
        if (!swapped) break;
        end -= 1;

        // Right to left pass
        swapped = false;
        var j: usize = end;
        while (j > start) : (j -= 1) {
            if (arr[j] < arr[j - 1]) {
                const tmp = arr[j];
                arr[j] = arr[j - 1];
                arr[j - 1] = tmp;
                swapped = true;
            }
        }
        if (!swapped) break;
        start += 1;
    }
}

test "cocktail shaker sort: basic case" {
    var arr = [_]i32{ 4, 5, 2, 1, 2 };
    cocktailShakerSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 2, 4, 5 }, &arr);
}

test "cocktail shaker sort: empty array" {
    var arr = [_]i32{};
    cocktailShakerSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "cocktail shaker sort: all negative" {
    var arr = [_]i32{ -4, -5, -24, -7, -11 };
    cocktailShakerSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -24, -11, -7, -5, -4 }, &arr);
}

test "cocktail shaker sort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    cocktailShakerSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "cocktail shaker sort: single element" {
    var arr = [_]i32{42};
    cocktailShakerSort(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}
