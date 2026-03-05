//! Max Difference Pair - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/max_difference_pair.py

const std = @import("std");
const testing = std.testing;

pub const MaxDiffError = error{EmptyInput};

pub const Pair = struct {
    small: i64,
    big: i64,
};

fn minValue(items: []const i64) i64 {
    var m = items[0];
    for (items[1..]) |v| m = @min(m, v);
    return m;
}

fn maxValue(items: []const i64) i64 {
    var m = items[0];
    for (items[1..]) |v| m = @max(m, v);
    return m;
}

/// Finds pair (small, big) maximizing (big - small) with ordering i <= j.
///
/// Time complexity: O(n log n)
/// Space complexity: O(log n) recursion depth
pub fn maxDifference(items: []const i64) MaxDiffError!Pair {
    if (items.len == 0) return MaxDiffError.EmptyInput;

    if (items.len == 1) {
        return .{ .small = items[0], .big = items[0] };
    }

    const mid = items.len / 2;
    const first = items[0..mid];
    const second = items[mid..];

    const pair1 = try maxDifference(first);
    const pair2 = try maxDifference(second);

    const min_first = minValue(first);
    const max_second = maxValue(second);
    const cross_diff = max_second - min_first;

    const diff2 = pair2.big - pair2.small;
    const diff1 = pair1.big - pair1.small;

    if (diff2 > cross_diff and diff2 > diff1) {
        return pair2;
    } else if (diff1 > cross_diff) {
        return pair1;
    }

    return .{ .small = min_first, .big = max_second };
}

test "max difference pair: python example" {
    const pair = try maxDifference(&[_]i64{ 5, 11, 2, 1, 7, 9, 0, 7 });
    try testing.expectEqual(@as(i64, 1), pair.small);
    try testing.expectEqual(@as(i64, 9), pair.big);
}

test "max difference pair: additional cases" {
    const p1 = try maxDifference(&[_]i64{ 5, 4, 3 });
    try testing.expectEqual(@as(i64, 5), p1.small);
    try testing.expectEqual(@as(i64, 5), p1.big);

    const p2 = try maxDifference(&[_]i64{1});
    try testing.expectEqual(@as(i64, 1), p2.small);
    try testing.expectEqual(@as(i64, 1), p2.big);

    const p3 = try maxDifference(&[_]i64{ 3, 2, 5, 1 });
    try testing.expectEqual(@as(i64, 2), p3.small);
    try testing.expectEqual(@as(i64, 5), p3.big);

    const p4 = try maxDifference(&[_]i64{ -5, -2, -7, -1 });
    try testing.expectEqual(@as(i64, -7), p4.small);
    try testing.expectEqual(@as(i64, -1), p4.big);
}

test "max difference pair: empty and extreme" {
    try testing.expectError(MaxDiffError.EmptyInput, maxDifference(&[_]i64{}));

    var arr: [1024]i64 = undefined;
    for (0..arr.len) |i| arr[i] = @intCast(i);
    const pair = try maxDifference(arr[0..]);
    try testing.expectEqual(@as(i64, 0), pair.small);
    try testing.expectEqual(@as(i64, 1023), pair.big);
}
