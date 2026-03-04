//! Bead Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/bead_sort.py

const std = @import("std");
const testing = std.testing;

pub const BeadSortError = error{NegativeValue};

/// In-place bead sort for non-negative integers.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn beadSort(sequence: []i64) BeadSortError!void {
    for (sequence) |x| {
        if (x < 0) return error.NegativeValue;
    }

    for (0..sequence.len) |_| {
        if (sequence.len <= 1) break;
        for (0..sequence.len - 1) |i| {
            const upper = sequence[i];
            const lower = sequence[i + 1];
            if (upper > lower) {
                const diff = upper - lower;
                sequence[i] -= diff;
                sequence[i + 1] += diff;
            }
        }
    }
}

test "bead sort: python reference examples" {
    var a1 = [_]i64{ 6, 11, 12, 4, 1, 5 };
    try beadSort(&a1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 4, 5, 6, 11, 12 }, &a1);

    var a2 = [_]i64{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try beadSort(&a2);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &a2);

    var a3 = [_]i64{ 5, 0, 4, 3 };
    try beadSort(&a3);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 3, 4, 5 }, &a3);
}

test "bead sort: invalid input and boundaries" {
    var a1 = [_]i64{ 8, 2, 1 };
    try beadSort(&a1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 8 }, &a1);

    var bad = [_]i64{ 1, 0, -1 };
    try testing.expectError(error.NegativeValue, beadSort(&bad));

    var empty = [_]i64{};
    try beadSort(&empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}

test "bead sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 3000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i64, @intCast(n - i));
    }

    try beadSort(arr);
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
    try testing.expectEqual(@as(i64, 1), arr[0]);
    try testing.expectEqual(@as(i64, @intCast(n)), arr[n - 1]);
}
