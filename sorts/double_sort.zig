//! Double Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/double_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place double-direction bubble-like sort.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn doubleSort(comptime T: type, arr: []T) void {
    const n = arr.len;
    if (n <= 1) return;

    const passes = ((n - 1) / 2) + 1;
    for (0..passes) |_| {
        var j: usize = 0;
        while (j < n - 1) : (j += 1) {
            if (arr[j + 1] < arr[j]) {
                std.mem.swap(T, &arr[j], &arr[j + 1]);
            }

            const right = n - 1 - j;
            if (arr[right] < arr[right - 1]) {
                std.mem.swap(T, &arr[right], &arr[right - 1]);
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

test "double sort: python reference examples" {
    var a1 = [_]i32{ -1, -2, -3, -4, -5, -6, -7 };
    doubleSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ -7, -6, -5, -4, -3, -2, -1 }, &a1);

    var a2 = [_]i32{};
    doubleSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ -1, -2, -3, -4, -5, -6 };
    doubleSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ -6, -5, -4, -3, -2, -1 }, &a3);

    var a4 = [_]i32{ -3, 10, 16, -42, 29 };
    doubleSort(i32, &a4);
    try testing.expectEqualSlices(i32, &[_]i32{ -42, -3, 10, 16, 29 }, &a4);
}

test "double sort: edge cases" {
    var one = [_]i32{5};
    doubleSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{5}, &one);

    var dup = [_]i32{ 2, 2, 2, 2 };
    doubleSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 2, 2, 2 }, &dup);
}

test "double sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 5000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    doubleSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
