//! Pigeonhole Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/pigeonhole_sort.py

const std = @import("std");
const testing = std.testing;

pub const PigeonholeSortError = std.mem.Allocator.Error || error{RangeTooLarge};

/// In-place pigeonhole sort for integers.
/// Time complexity: O(n + range), Space complexity: O(range)
pub fn pigeonholeSort(allocator: std.mem.Allocator, arr: []i64) PigeonholeSortError!void {
    if (arr.len == 0) return;

    var min_val = arr[0];
    var max_val = arr[0];
    for (arr[1..]) |x| {
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }

    const size_i128 = @as(i128, max_val) - @as(i128, min_val) + 1;
    if (size_i128 <= 0 or size_i128 > std.math.maxInt(usize)) return error.RangeTooLarge;
    const size: usize = @intCast(size_i128);

    const holes = try allocator.alloc(usize, size);
    defer allocator.free(holes);
    @memset(holes, 0);

    for (arr) |x| {
        holes[@intCast(x - min_val)] += 1;
    }

    var i: usize = 0;
    for (0..size) |count| {
        while (holes[count] > 0) {
            holes[count] -= 1;
            arr[i] = @as(i64, @intCast(count)) + min_val;
            i += 1;
        }
    }
}

test "pigeonhole sort: python reference example" {
    const alloc = testing.allocator;
    var a = [_]i64{ 8, 3, 2, 7, 4, 6, 8 };
    try pigeonholeSort(alloc, &a);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 3, 4, 6, 7, 8, 8 }, &a);
}

test "pigeonhole sort: edge cases" {
    const alloc = testing.allocator;

    var one = [_]i64{5};
    try pigeonholeSort(alloc, &one);
    try testing.expectEqualSlices(i64, &[_]i64{5}, &one);

    var empty = [_]i64{};
    try pigeonholeSort(alloc, &empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var mixed = [_]i64{ -2, 5, 0, -4 };
    try pigeonholeSort(alloc, &mixed);
    try testing.expectEqualSlices(i64, &[_]i64{ -4, -2, 0, 5 }, &mixed);
}

test "pigeonhole sort: extreme bounded-range input" {
    const alloc = testing.allocator;
    const n: usize = 30_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, idx| {
        // Keep value range compact for memory-safe pigeonholes.
        v.* = @as(i64, @intCast((idx * 17) % 2001)) - 1000;
    }

    try pigeonholeSort(alloc, arr);
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}
