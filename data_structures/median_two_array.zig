//! Median Two Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/median_two_array.py

const std = @import("std");
const testing = std.testing;

/// Finds median of two arrays by merge-and-sort strategy.
/// Time complexity: O((n+m) log(n+m)), Space complexity: O(n+m)
pub fn findMedianSortedArrays(allocator: std.mem.Allocator, nums1: []const f64, nums2: []const f64) !f64 {
    if (nums1.len == 0 and nums2.len == 0) return error.BothInputArraysEmpty;

    const total = nums1.len + nums2.len;
    const merged = try allocator.alloc(f64, total);
    defer allocator.free(merged);

    @memcpy(merged[0..nums1.len], nums1);
    @memcpy(merged[nums1.len..], nums2);
    std.mem.sort(f64, merged, {}, std.sort.asc(f64));

    if (total % 2 == 1) {
        return merged[total / 2];
    }

    const middle1 = merged[total / 2 - 1];
    const middle2 = merged[total / 2];
    return (middle1 + middle2) / 2.0;
}

test "median two array: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.0), try findMedianSortedArrays(testing.allocator, &[_]f64{ 1, 3 }, &[_]f64{2}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.5), try findMedianSortedArrays(testing.allocator, &[_]f64{ 1, 2 }, &[_]f64{ 3, 4 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try findMedianSortedArrays(testing.allocator, &[_]f64{ 0, 0 }, &[_]f64{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try findMedianSortedArrays(testing.allocator, &[_]f64{}, &[_]f64{1}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try findMedianSortedArrays(testing.allocator, &[_]f64{-1000}, &[_]f64{1000}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -2.75), try findMedianSortedArrays(testing.allocator, &[_]f64{ -1.1, -2.2 }, &[_]f64{ -3.3, -4.4 }), 1e-9);
}

test "median two array: invalid and extreme" {
    try testing.expectError(error.BothInputArraysEmpty, findMedianSortedArrays(testing.allocator, &[_]f64{}, &[_]f64{}));

    const n: usize = 50_000;
    const left = try testing.allocator.alloc(f64, n);
    defer testing.allocator.free(left);
    const right = try testing.allocator.alloc(f64, n);
    defer testing.allocator.free(right);

    for (0..n) |i| {
        left[i] = @floatFromInt(i);
        right[i] = @floatFromInt(n + i);
    }

    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(n)) - 0.5, try findMedianSortedArrays(testing.allocator, left, right), 1e-9);
}
