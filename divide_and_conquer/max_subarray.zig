//! Maximum Subarray (Divide and Conquer) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/max_subarray.py

const std = @import("std");
const testing = std.testing;

pub const MaxSubarrayError = error{ InvalidRange, Overflow };

pub const MaxSubarrayResult = struct {
    start: ?usize,
    end: ?usize,
    sum: i64,
};

fn addChecked(a: i64, b: i64) MaxSubarrayError!i64 {
    const added = @addWithOverflow(a, b);
    if (added[1] != 0) return MaxSubarrayError.Overflow;
    return added[0];
}

fn maxCrossSum(items: []const i64, low: usize, mid: usize, high: usize) MaxSubarrayError!MaxSubarrayResult {
    var left_sum: i64 = std.math.minInt(i64);
    var max_left = low;
    var sum_left: i64 = 0;

    var i = mid;
    while (true) {
        sum_left = try addChecked(sum_left, items[i]);
        if (sum_left > left_sum) {
            left_sum = sum_left;
            max_left = i;
        }

        if (i == low) break;
        i -= 1;
    }

    var right_sum: i64 = std.math.minInt(i64);
    var max_right = mid + 1;
    var sum_right: i64 = 0;

    var j = mid + 1;
    while (j <= high) : (j += 1) {
        sum_right = try addChecked(sum_right, items[j]);
        if (sum_right > right_sum) {
            right_sum = sum_right;
            max_right = j;
        }
    }

    return .{
        .start = max_left,
        .end = max_right,
        .sum = try addChecked(left_sum, right_sum),
    };
}

/// Finds the maximum-sum contiguous subarray in items[low..high].
/// For empty input, returns {null, null, 0}.
///
/// Time complexity: O(n log n)
/// Space complexity: O(log n) recursion stack
pub fn maxSubarray(items: []const i64, low: usize, high: usize) MaxSubarrayError!MaxSubarrayResult {
    if (items.len == 0) {
        return .{ .start = null, .end = null, .sum = 0 };
    }

    if (low > high or high >= items.len) return MaxSubarrayError.InvalidRange;

    if (low == high) {
        return .{ .start = low, .end = high, .sum = items[low] };
    }

    const mid = low + (high - low) / 2;
    const left = try maxSubarray(items, low, mid);
    const right = try maxSubarray(items, mid + 1, high);
    const cross = try maxCrossSum(items, low, mid, high);

    if (left.sum >= right.sum and left.sum >= cross.sum) return left;
    if (right.sum >= left.sum and right.sum >= cross.sum) return right;
    return cross;
}

test "max subarray divide and conquer: mixed input" {
    const nums = [_]i64{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try maxSubarray(&nums, 0, nums.len - 1);

    try testing.expectEqual(@as(?usize, 3), result.start);
    try testing.expectEqual(@as(?usize, 6), result.end);
    try testing.expectEqual(@as(i64, 6), result.sum);
}

test "max subarray divide and conquer: all positive" {
    const nums = [_]i64{ 2, 8, 9 };
    const result = try maxSubarray(&nums, 0, nums.len - 1);

    try testing.expectEqual(@as(?usize, 0), result.start);
    try testing.expectEqual(@as(?usize, 2), result.end);
    try testing.expectEqual(@as(i64, 19), result.sum);
}

test "max subarray divide and conquer: all negative" {
    const nums = [_]i64{ -2, -3, -1, -4, -6 };
    const result = try maxSubarray(&nums, 0, nums.len - 1);

    try testing.expectEqual(@as(?usize, 2), result.start);
    try testing.expectEqual(@as(?usize, 2), result.end);
    try testing.expectEqual(@as(i64, -1), result.sum);
}

test "max subarray divide and conquer: tie follows left branch" {
    const nums = [_]i64{ 0, 0 };
    const result = try maxSubarray(&nums, 0, nums.len - 1);

    try testing.expectEqual(@as(?usize, 0), result.start);
    try testing.expectEqual(@as(?usize, 0), result.end);
    try testing.expectEqual(@as(i64, 0), result.sum);
}

test "max subarray divide and conquer: empty and invalid range" {
    const empty = [_]i64{};
    const empty_result = try maxSubarray(&empty, 0, 0);
    try testing.expectEqual(@as(?usize, null), empty_result.start);
    try testing.expectEqual(@as(?usize, null), empty_result.end);
    try testing.expectEqual(@as(i64, 0), empty_result.sum);

    const nums = [_]i64{ 1, 2, 3 };
    try testing.expectError(MaxSubarrayError.InvalidRange, maxSubarray(&nums, 2, 1));
    try testing.expectError(MaxSubarrayError.InvalidRange, maxSubarray(&nums, 0, 3));
}

test "max subarray divide and conquer: overflow is reported" {
    const nums = [_]i64{ std.math.maxInt(i64), 1 };
    try testing.expectError(MaxSubarrayError.Overflow, maxSubarray(&nums, 0, nums.len - 1));
}
