//! Minimum Size Subarray Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_size_subarray_sum.py

const std = @import("std");
const testing = std.testing;

pub const MinimumSubarraySumError = error{
    Overflow,
};

/// Returns the minimal length of a contiguous subarray with sum >= target.
/// Returns 0 if no such subarray exists.
/// Time complexity: O(n), Space complexity: O(1)
pub fn minimumSubarraySum(
    target: i64,
    numbers: []const i64,
) MinimumSubarraySumError!usize {
    if (numbers.len == 0) return 0;

    if (target == 0) {
        for (numbers) |value| {
            if (value == 0) return 0;
        }
    }

    var left: usize = 0;
    var right: usize = 0;
    var current_sum: i64 = 0;
    var min_len: usize = std.math.maxInt(usize);

    while (right < numbers.len) : (right += 1) {
        const add = @addWithOverflow(current_sum, numbers[right]);
        if (add[1] != 0) return MinimumSubarraySumError.Overflow;
        current_sum = add[0];

        while (current_sum >= target and left <= right) {
            const candidate = right - left + 1;
            if (candidate < min_len) min_len = candidate;

            const sub = @subWithOverflow(current_sum, numbers[left]);
            if (sub[1] != 0) return MinimumSubarraySumError.Overflow;
            current_sum = sub[0];
            left += 1;
        }
    }

    return if (min_len == std.math.maxInt(usize)) 0 else min_len;
}

test "minimum size subarray sum: python examples" {
    try testing.expectEqual(@as(usize, 2), try minimumSubarraySum(7, &[_]i64{ 2, 3, 1, 2, 4, 3 }));
    try testing.expectEqual(@as(usize, 4), try minimumSubarraySum(7, &[_]i64{ 2, 3, -1, 2, 4, -3 }));
    try testing.expectEqual(@as(usize, 0), try minimumSubarraySum(11, &[_]i64{ 1, 1, 1, 1, 1, 1, 1, 1 }));
    try testing.expectEqual(@as(usize, 2), try minimumSubarraySum(10, &[_]i64{ 1, 2, 3, 4, 5, 6, 7 }));
    try testing.expectEqual(@as(usize, 1), try minimumSubarraySum(5, &[_]i64{ 1, 1, 1, 1, 1, 5 }));
    try testing.expectEqual(@as(usize, 0), try minimumSubarraySum(0, &[_]i64{}));
    try testing.expectEqual(@as(usize, 1), try minimumSubarraySum(0, &[_]i64{ 1, 2, 3 }));
    try testing.expectEqual(@as(usize, 1), try minimumSubarraySum(10, &[_]i64{ 10, 20, 30 }));
    try testing.expectEqual(@as(usize, 1), try minimumSubarraySum(7, &[_]i64{ 1, 1, 1, 1, 1, 1, 10 }));
    try testing.expectEqual(@as(usize, 0), try minimumSubarraySum(6, &[_]i64{}));
    try testing.expectEqual(@as(usize, 1), try minimumSubarraySum(2, &[_]i64{ 1, 2, 3 }));
    try testing.expectEqual(@as(usize, 0), try minimumSubarraySum(-6, &[_]i64{}));
    try testing.expectEqual(@as(usize, 1), try minimumSubarraySum(-6, &[_]i64{ 3, 4, 5 }));
}

test "minimum size subarray sum: target zero with zero inside" {
    try testing.expectEqual(@as(usize, 0), try minimumSubarraySum(0, &[_]i64{ 1, 0, 3 }));
}

test "minimum size subarray sum: extreme long positive array" {
    var arr: [5000]i64 = undefined;
    @memset(&arr, 1);
    try testing.expectEqual(@as(usize, 4096), try minimumSubarraySum(4096, &arr));
}

test "minimum size subarray sum: overflow detection" {
    const arr = [_]i64{ std.math.maxInt(i64) - 1, 10 };
    try testing.expectError(MinimumSubarraySumError.Overflow, minimumSubarraySum(std.math.maxInt(i64), &arr));
}
