//! Find Min - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/find_min.py

const std = @import("std");
const testing = std.testing;

pub const FindMinError = error{ EmptyInput, IndexOutOfRange };

/// Finds minimum value iteratively.
/// Time complexity: O(n), Space complexity: O(1)
pub fn findMinIterative(nums: []const f64) FindMinError!f64 {
    if (nums.len == 0) return FindMinError.EmptyInput;

    var min_value = nums[0];
    for (nums) |value| {
        if (value < min_value) min_value = value;
    }
    return min_value;
}

/// Finds minimum value recursively on index range [left, right] (supports negative indices).
/// Time complexity: O(n), Space complexity: O(log n)
pub fn findMinRecursive(nums: []const f64, left: i64, right: i64) FindMinError!f64 {
    if (nums.len == 0) return FindMinError.EmptyInput;

    const l = try normalizeIndex(nums.len, left);
    const r = try normalizeIndex(nums.len, right);
    if (l > r) return FindMinError.IndexOutOfRange;
    return findMinRecursiveRange(nums, l, r);
}

fn normalizeIndex(len: usize, idx: i64) FindMinError!usize {
    const len_i64: i64 = @intCast(len);
    if (idx >= len_i64 or idx < -len_i64) return FindMinError.IndexOutOfRange;
    if (idx >= 0) return @intCast(idx);
    return @intCast(len_i64 + idx);
}

fn findMinRecursiveRange(nums: []const f64, left: usize, right: usize) f64 {
    if (left == right) return nums[left];
    const mid = (left + right) >> 1;
    const left_min = findMinRecursiveRange(nums, left, mid);
    const right_min = findMinRecursiveRange(nums, mid + 1, right);
    return if (left_min <= right_min) left_min else right_min;
}

test "find min: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, -56), try findMinIterative(&[_]f64{ 0, 1, 2, 3, 4, 5, -3, 24, -56 }), 1e-12);

    const nums = [_]f64{ 1, 3, 5, 7, 9, 2, 4, 6, 8, 10 };
    try testing.expectApproxEqAbs(@as(f64, 1), try findMinRecursive(&nums, 0, @intCast(nums.len - 1)), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), try findMinRecursive(&nums, -@as(i64, @intCast(nums.len)), -1), 1e-12);
}

test "find min: invalid and edge cases" {
    try testing.expectError(FindMinError.EmptyInput, findMinIterative(&[_]f64{}));
    try testing.expectError(FindMinError.EmptyInput, findMinRecursive(&[_]f64{}, 0, 0));

    const nums = [_]f64{ 1, 2, 3, 4 };
    try testing.expectError(FindMinError.IndexOutOfRange, findMinRecursive(&nums, 0, @intCast(nums.len)));
    try testing.expectError(FindMinError.IndexOutOfRange, findMinRecursive(&nums, -@as(i64, @intCast(nums.len)) - 1, -1));
    try testing.expectError(FindMinError.IndexOutOfRange, findMinRecursive(&nums, 3, 1));
}
