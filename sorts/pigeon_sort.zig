//! Pigeon Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/pigeon_sort.py

const std = @import("std");
const testing = std.testing;

pub const PigeonSortError = std.mem.Allocator.Error || error{RangeTooLarge};

/// In-place pigeon sort for integers.
/// Time complexity: O(n + range), Space complexity: O(range)
pub fn pigeonSort(allocator: std.mem.Allocator, arr: []i64) PigeonSortError!void {
    if (arr.len == 0) return;

    var min_val = arr[0];
    var max_val = arr[0];
    for (arr[1..]) |v| {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    const range_i128 = @as(i128, max_val) - @as(i128, min_val) + 1;
    if (range_i128 <= 0 or range_i128 > std.math.maxInt(usize)) return error.RangeTooLarge;
    const range: usize = @intCast(range_i128);

    const holes_repeat = try allocator.alloc(usize, range);
    defer allocator.free(holes_repeat);
    @memset(holes_repeat, 0);

    for (arr) |v| {
        const index: usize = @intCast(v - min_val);
        holes_repeat[index] += 1;
    }

    var write_index: usize = 0;
    for (0..range) |i| {
        while (holes_repeat[i] > 0) {
            arr[write_index] = min_val + @as(i64, @intCast(i));
            write_index += 1;
            holes_repeat[i] -= 1;
        }
    }
}

test "pigeon sort: python reference examples" {
    const alloc = testing.allocator;

    var a1 = [_]i64{ 0, 5, 3, 2, 2 };
    try pigeonSort(alloc, &a1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 2, 2, 3, 5 }, &a1);

    var a2 = [_]i64{};
    try pigeonSort(alloc, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i64{ -2, -5, -45 };
    try pigeonSort(alloc, &a3);
    try testing.expectEqualSlices(i64, &[_]i64{ -45, -5, -2 }, &a3);
}

test "pigeon sort: edge cases" {
    const alloc = testing.allocator;

    var one = [_]i64{8};
    try pigeonSort(alloc, &one);
    try testing.expectEqualSlices(i64, &[_]i64{8}, &one);

    var dup = [_]i64{ 4, 4, 4, 4 };
    try pigeonSort(alloc, &dup);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 4, 4, 4 }, &dup);
}

test "pigeon sort: extreme mixed input" {
    const alloc = testing.allocator;
    const n: usize = 25_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        // Bounded range for memory-safe pigeon holes.
        const value = @as(i64, @intCast((i % 4001))) - 2000;
        v.* = if ((i % 2) == 0) value else -value;
    }

    try pigeonSort(alloc, arr);
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}
