//! Max Sum Sliding Window - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/max_sum_sliding_window.py

const std = @import("std");
const testing = std.testing;

pub const SlidingWindowError = error{ InvalidInput, Overflow };

/// Returns maximum sum of `k` consecutive elements.
/// Time complexity: O(n), Space complexity: O(1)
pub fn maxSumInArray(array: []const i64, k: i64) SlidingWindowError!i64 {
    if (k < 0) return SlidingWindowError.InvalidInput;

    const k_usize: usize = @intCast(k);
    if (array.len < k_usize) return SlidingWindowError.InvalidInput;
    if (k_usize == 0) return 0;

    var current_sum: i128 = 0;
    for (array[0..k_usize]) |value| current_sum += value;
    var max_sum = current_sum;

    var i: usize = 0;
    while (i < array.len - k_usize) : (i += 1) {
        current_sum = current_sum - array[i] + array[i + k_usize];
        if (current_sum > max_sum) max_sum = current_sum;
    }

    if (max_sum > std.math.maxInt(i64) or max_sum < std.math.minInt(i64)) return SlidingWindowError.Overflow;
    return @intCast(max_sum);
}

test "max sum sliding window: python reference examples" {
    try testing.expectEqual(@as(i64, 24), try maxSumInArray(&[_]i64{ 1, 4, 2, 10, 2, 3, 1, 0, 20 }, 4));
    try testing.expectError(SlidingWindowError.InvalidInput, maxSumInArray(&[_]i64{ 1, 4, 2, 10, 2, 3, 1, 0, 20 }, 10));
    try testing.expectEqual(@as(i64, 27), try maxSumInArray(&[_]i64{ 1, 4, 2, 10, 2, 13, 1, 0, 2 }, 4));
}

test "max sum sliding window: edge and extreme cases" {
    try testing.expectEqual(@as(i64, 0), try maxSumInArray(&[_]i64{ 1, 2, 3 }, 0));
    try testing.expectError(SlidingWindowError.InvalidInput, maxSumInArray(&[_]i64{ 1, 2, 3 }, -1));
    try testing.expectEqual(@as(i64, 6), try maxSumInArray(&[_]i64{ 1, 2, 3 }, 3));
}
