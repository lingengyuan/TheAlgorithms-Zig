//! Pancake Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/pancake_sort.py

const std = @import("std");
const testing = std.testing;

fn reversePrefix(comptime T: type, arr: []T, last_idx: usize) void {
    if (arr.len == 0) return;
    var l: usize = 0;
    var r: usize = last_idx;
    while (l < r) {
        std.mem.swap(T, &arr[l], &arr[r]);
        l += 1;
        r -= 1;
    }
}

fn maxIndex(comptime T: type, arr: []const T, end: usize) usize {
    var mi: usize = 0;
    for (1..end) |i| {
        if (arr[i] > arr[mi]) mi = i;
    }
    return mi;
}

/// In-place pancake sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn pancakeSort(comptime T: type, arr: []T) void {
    var cur = arr.len;
    while (cur > 1) : (cur -= 1) {
        const mi = maxIndex(T, arr, cur);
        if (mi == cur - 1) continue;
        reversePrefix(T, arr, mi);
        reversePrefix(T, arr, cur - 1);
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "pancake sort: python reference examples" {
    var a1 = [_]i32{ 0, 5, 3, 2, 2 };
    pancakeSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &a1);

    var a2 = [_]i32{};
    pancakeSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ -2, -5, -45 };
    pancakeSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &a3);
}

test "pancake sort: edge cases" {
    var one = [_]i32{42};
    pancakeSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &one);

    var dup = [_]i32{ 3, 1, 3, 1, 3, 1 };
    pancakeSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 1, 3, 3, 3 }, &dup);
}

test "pancake sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 3000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    pancakeSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
