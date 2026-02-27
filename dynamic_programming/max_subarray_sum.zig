//! Maximum Subarray Sum (Kadane) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/max_subarray_sum.py

const std = @import("std");
const testing = std.testing;

/// Returns the maximum sum over all non-empty contiguous subarrays.
/// Returns null for an empty slice.
/// Time complexity: O(n), Space complexity: O(1)
pub fn maxSubarraySum(items: []const i64) ?i64 {
    if (items.len == 0) return null;

    var best = items[0];
    var current = items[0];

    for (items[1..]) |value| {
        current = @max(value, current + value);
        best = @max(best, current);
    }

    return best;
}

test "max subarray sum: mixed values" {
    const arr = [_]i64{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    try testing.expectEqual(@as(?i64, 6), maxSubarraySum(&arr));
}

test "max subarray sum: all negative" {
    const arr = [_]i64{ -8, -3, -6, -2, -5, -4 };
    try testing.expectEqual(@as(?i64, -2), maxSubarraySum(&arr));
}

test "max subarray sum: all positive" {
    const arr = [_]i64{ 1, 2, 3, 4 };
    try testing.expectEqual(@as(?i64, 10), maxSubarraySum(&arr));
}

test "max subarray sum: single element" {
    const arr = [_]i64{42};
    try testing.expectEqual(@as(?i64, 42), maxSubarraySum(&arr));
}

test "max subarray sum: empty array" {
    const arr = [_]i64{};
    try testing.expectEqual(@as(?i64, null), maxSubarraySum(&arr));
}
