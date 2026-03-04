//! Maximum Non-Adjacent Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/max_non_adjacent_sum.py

const std = @import("std");
const testing = std.testing;

pub const MaxNonAdjacentSumError = error{
    Overflow,
};

/// Returns the maximum sum of non-adjacent elements.
/// Matches Python behavior for empty/all-negative arrays (returns 0).
/// Time complexity: O(n), Space complexity: O(1)
pub fn maximumNonAdjacentSum(numbers: []const i64) MaxNonAdjacentSumError!i64 {
    if (numbers.len == 0) return 0;

    var max_including = numbers[0];
    var max_excluding: i64 = 0;

    for (numbers[1..]) |num| {
        const include_next = @addWithOverflow(max_excluding, num);
        if (include_next[1] != 0) return MaxNonAdjacentSumError.Overflow;
        const exclude_next = @max(max_including, max_excluding);

        max_including = include_next[0];
        max_excluding = exclude_next;
    }

    return @max(max_excluding, max_including);
}

test "maximum non adjacent sum: python examples" {
    try testing.expectEqual(@as(i64, 4), try maximumNonAdjacentSum(&[_]i64{ 1, 2, 3 }));
    try testing.expectEqual(@as(i64, 18), try maximumNonAdjacentSum(&[_]i64{ 1, 5, 3, 7, 2, 2, 6 }));
    try testing.expectEqual(@as(i64, 0), try maximumNonAdjacentSum(&[_]i64{ -1, -5, -3, -7, -2, -2, -6 }));
    try testing.expectEqual(@as(i64, 500), try maximumNonAdjacentSum(&[_]i64{ 499, 500, -3, -7, -2, -2, -6 }));
}

test "maximum non adjacent sum: boundary cases" {
    try testing.expectEqual(@as(i64, 0), try maximumNonAdjacentSum(&[_]i64{}));
    try testing.expectEqual(@as(i64, 9), try maximumNonAdjacentSum(&[_]i64{9}));
    try testing.expectEqual(@as(i64, 9), try maximumNonAdjacentSum(&[_]i64{ 9, 8 }));
    try testing.expectEqual(@as(i64, 11), try maximumNonAdjacentSum(&[_]i64{ 2, 1, 4, 9, 2 }));
}

test "maximum non adjacent sum: extreme long alternating sequence" {
    var values: [4096]i64 = undefined;
    for (0..values.len) |i| {
        values[i] = if ((i % 2) == 0) 1 else -1;
    }
    try testing.expectEqual(@as(i64, 2048), try maximumNonAdjacentSum(&values));
}

test "maximum non adjacent sum: overflow detection" {
    const values = [_]i64{ std.math.maxInt(i64), 0, std.math.maxInt(i64) };
    try testing.expectError(MaxNonAdjacentSumError.Overflow, maximumNonAdjacentSum(&values));
}
