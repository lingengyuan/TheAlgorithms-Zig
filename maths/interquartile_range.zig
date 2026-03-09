//! Interquartile Range - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/interquartile_range.py

const std = @import("std");
const testing = std.testing;

pub const InterquartileRangeError = error{EmptyList};

fn lessThan(_: void, lhs: f64, rhs: f64) bool {
    return lhs < rhs;
}

/// Returns the median of a sorted slice.
/// Time complexity: O(1), Space complexity: O(1)
pub fn findMedian(nums: []const f64) f64 {
    if (nums.len == 0) return 0.0;
    const div = nums.len / 2;
    if (nums.len % 2 == 1) return nums[div];
    return (nums[div] + nums[div - 1]) / 2.0;
}

/// Returns the interquartile range of a numeric list.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn interquartileRange(allocator: std.mem.Allocator, nums: []const f64) (InterquartileRangeError || std.mem.Allocator.Error)!f64 {
    if (nums.len == 0) return error.EmptyList;

    const sorted = try allocator.dupe(f64, nums);
    defer allocator.free(sorted);
    std.sort.heap(f64, sorted, {}, lessThan);

    const div = sorted.len / 2;
    const mod = sorted.len % 2;
    const q1 = findMedian(sorted[0..div]);
    const half_length = div + mod;
    const q3 = findMedian(sorted[half_length..]);
    return q3 - q1;
}

test "interquartile range: python reference examples" {
    const alloc = testing.allocator;
    try testing.expectApproxEqAbs(@as(f64, 2.0), try interquartileRange(alloc, &[_]f64{ 4, 1, 2, 3, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 17.0), try interquartileRange(alloc, &[_]f64{ -2, -7, -10, 9, 8, 4, -67, 45 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 17.2), try interquartileRange(alloc, &[_]f64{ -2.1, -7.1, -10.1, 9.1, 8.1, 4.1, -67.1, 45.1 }), 1e-12);
}

test "interquartile range: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectApproxEqAbs(@as(f64, 0.0), try interquartileRange(alloc, &[_]f64{ 0, 0, 0, 0, 0 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try interquartileRange(alloc, &[_]f64{5}), 1e-12);
    try testing.expectError(error.EmptyList, interquartileRange(alloc, &[_]f64{}));
}
