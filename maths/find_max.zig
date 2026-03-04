//! Find Max - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/find_max.py

const std = @import("std");
const testing = std.testing;

pub const FindMaxError = error{ EmptyInput, IndexOutOfRange };

/// Finds maximum value iteratively.
/// Time complexity: O(n), Space complexity: O(1)
pub fn findMaxIterative(nums: []const f64) FindMaxError!f64 {
    if (nums.len == 0) return FindMaxError.EmptyInput;

    var max_value = nums[0];
    for (nums) |value| {
        if (value > max_value) max_value = value;
    }
    return max_value;
}

/// Finds maximum value recursively on index range [left, right] (supports negative indices).
/// Time complexity: O(n), Space complexity: O(log n)
pub fn findMaxRecursive(nums: []const f64, left: i64, right: i64) FindMaxError!f64 {
    if (nums.len == 0) return FindMaxError.EmptyInput;

    const l = try normalizeIndex(nums.len, left);
    const r = try normalizeIndex(nums.len, right);
    if (l > r) return FindMaxError.IndexOutOfRange;
    return findMaxRecursiveRange(nums, l, r);
}

fn normalizeIndex(len: usize, idx: i64) FindMaxError!usize {
    const len_i64: i64 = @intCast(len);
    if (idx >= len_i64 or idx < -len_i64) return FindMaxError.IndexOutOfRange;
    if (idx >= 0) return @intCast(idx);
    return @intCast(len_i64 + idx);
}

fn findMaxRecursiveRange(nums: []const f64, left: usize, right: usize) f64 {
    if (left == right) return nums[left];
    const mid = (left + right) >> 1;
    const left_max = findMaxRecursiveRange(nums, left, mid);
    const right_max = findMaxRecursiveRange(nums, mid + 1, right);
    return if (left_max >= right_max) left_max else right_max;
}

test "find max: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 94), try findMaxIterative(&[_]f64{ 2, 4, 9, 7, 19, 94, 5 }), 1e-12);

    const nums = [_]f64{ 1, 3, 5, 7, 9, 2, 4, 6, 8, 10 };
    try testing.expectApproxEqAbs(@as(f64, 10), try findMaxRecursive(&nums, 0, @intCast(nums.len - 1)), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10), try findMaxRecursive(&nums, -@as(i64, @intCast(nums.len)), -1), 1e-12);
}

test "find max: invalid and edge cases" {
    try testing.expectError(FindMaxError.EmptyInput, findMaxIterative(&[_]f64{}));
    try testing.expectError(FindMaxError.EmptyInput, findMaxRecursive(&[_]f64{}, 0, 0));

    const nums = [_]f64{ 1, 2, 3, 4 };
    try testing.expectError(FindMaxError.IndexOutOfRange, findMaxRecursive(&nums, 0, @intCast(nums.len)));
    try testing.expectError(FindMaxError.IndexOutOfRange, findMaxRecursive(&nums, -@as(i64, @intCast(nums.len)) - 1, -1));
    try testing.expectError(FindMaxError.IndexOutOfRange, findMaxRecursive(&nums, 3, 1));
}
