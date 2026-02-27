//! Maximum Subarray Sum (Kadane) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/max_subarray_sum.py

const std = @import("std");
const testing = std.testing;

/// Returns the maximum sum over all non-empty contiguous subarrays.
/// Time complexity: O(n), Space complexity: O(1)
pub fn maxSubarraySum(items: []const i64, allow_empty_subarrays: bool) i64 {
    if (items.len == 0) return 0;

    var best: i64 = if (allow_empty_subarrays) 0 else std.math.minInt(i64);
    var current: i64 = 0;

    for (items) |value| {
        current = @max(if (allow_empty_subarrays) 0 else value, current + value);
        best = @max(best, current);
    }

    return best;
}

test "max subarray sum: mixed values" {
    const arr = [_]i64{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    try testing.expectEqual(@as(i64, 6), maxSubarraySum(&arr, false));
}

test "max subarray sum: all negative" {
    const arr = [_]i64{ -2, -3, -1, -4, -6 };
    try testing.expectEqual(@as(i64, -1), maxSubarraySum(&arr, false));
}

test "max subarray sum: all positive" {
    const arr = [_]i64{ 1, 2, 3, 4 };
    try testing.expectEqual(@as(i64, 10), maxSubarraySum(&arr, false));
}

test "max subarray sum: single element" {
    const arr = [_]i64{42};
    try testing.expectEqual(@as(i64, 42), maxSubarraySum(&arr, false));
}

test "max subarray sum: empty array" {
    const arr = [_]i64{};
    try testing.expectEqual(@as(i64, 0), maxSubarraySum(&arr, false));
}

test "max subarray sum: allow empty subarrays" {
    const arr = [_]i64{ -2, -3, -1, -4, -6 };
    try testing.expectEqual(@as(i64, 0), maxSubarraySum(&arr, true));
}
