//! Comb Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/comb_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place comb sort, ascending order.
/// Time complexity: O(n²) worst, better than bubble sort on average
/// Space complexity: O(1)
pub fn combSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var gap = arr.len;
    var completed = false;

    while (!completed) {
        gap = (gap * 10) / 13;
        if (gap <= 1) {
            gap = 1;
            completed = true;
        }

        var index: usize = 0;
        while (index + gap < arr.len) : (index += 1) {
            if (arr[index] > arr[index + gap]) {
                std.mem.swap(T, &arr[index], &arr[index + gap]);
                completed = false;
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

test "comb sort: python reference examples" {
    var a1 = [_]i32{ 0, 5, 3, 2, 2 };
    combSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &a1);

    var a2 = [_]i32{};
    combSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ 99, 45, -7, 8, 2, 0, -15, 3 };
    combSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ -15, -7, 0, 2, 3, 8, 45, 99 }, &a3);
}

test "comb sort: edge cases" {
    var sorted = [_]i32{ 1, 2, 3, 4 };
    combSort(i32, &sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, &sorted);

    var dup = [_]i32{ 7, 7, 7, 7 };
    combSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7 }, &dup);
}

test "comb sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 6000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    combSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
