//! Odd-Even Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/odd_even_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place odd-even sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn oddEvenSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var is_sorted = false;
    while (!is_sorted) {
        is_sorted = true;

        var i: usize = 0;
        while (i + 1 < arr.len) : (i += 2) {
            if (arr[i] > arr[i + 1]) {
                std.mem.swap(T, &arr[i], &arr[i + 1]);
                is_sorted = false;
            }
        }

        i = 1;
        while (i + 1 < arr.len) : (i += 2) {
            if (arr[i] > arr[i + 1]) {
                std.mem.swap(T, &arr[i], &arr[i + 1]);
                is_sorted = false;
            }
        }
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "odd even sort: python reference examples" {
    var a1 = [_]i32{ 5, 4, 3, 2, 1 };
    oddEvenSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &a1);

    var a2 = [_]i32{};
    oddEvenSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ -10, -1, 10, 2 };
    oddEvenSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ -10, -1, 2, 10 }, &a3);

    var a4 = [_]i32{ 1, 2, 3, 4 };
    oddEvenSort(i32, &a4);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, &a4);
}

test "odd even sort: edge cases" {
    var one = [_]i32{5};
    oddEvenSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{5}, &one);

    var dup = [_]i32{ 3, 3, 3, 3, 3 };
    oddEvenSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 3, 3, 3, 3 }, &dup);
}

test "odd even sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 5000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    oddEvenSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
