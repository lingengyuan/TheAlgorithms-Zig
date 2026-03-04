//! Exchange Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/exchange_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place exchange sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn exchangeSort(comptime T: type, arr: []T) void {
    const n = arr.len;
    for (0..n) |i| {
        for (i + 1..n) |j| {
            if (arr[j] < arr[i]) {
                std.mem.swap(T, &arr[i], &arr[j]);
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

test "exchange sort: python reference examples" {
    var a1 = [_]i32{ 5, 4, 3, 2, 1 };
    exchangeSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &a1);

    var a2 = [_]i32{ -1, -2, -3 };
    exchangeSort(i32, &a2);
    try testing.expectEqualSlices(i32, &[_]i32{ -3, -2, -1 }, &a2);

    var a3 = [_]i32{ 1, 2, 3, 4, 5 };
    exchangeSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &a3);

    var a4 = [_]i32{ 0, 10, -2, 5, 3 };
    exchangeSort(i32, &a4);
    try testing.expectEqualSlices(i32, &[_]i32{ -2, 0, 3, 5, 10 }, &a4);
}

test "exchange sort: edge cases" {
    var empty = [_]i32{};
    exchangeSort(i32, &empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var dup = [_]i32{ 2, 2, 2, 2 };
    exchangeSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 2, 2, 2 }, &dup);
}

test "exchange sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 4000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    exchangeSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
