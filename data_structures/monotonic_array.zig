//! Monotonic Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/monotonic_array.py

const std = @import("std");
const testing = std.testing;

/// Returns true if array is monotonic (non-decreasing or non-increasing).
/// Time complexity: O(n), Space complexity: O(1)
pub fn isMonotonic(nums: []const i64) bool {
    if (nums.len <= 2) return true;

    var non_decreasing = true;
    var non_increasing = true;

    var i: usize = 0;
    while (i + 1 < nums.len) : (i += 1) {
        if (nums[i] > nums[i + 1]) non_decreasing = false;
        if (nums[i] < nums[i + 1]) non_increasing = false;
        if (!non_decreasing and !non_increasing) return false;
    }

    return true;
}

test "monotonic array: python examples" {
    try testing.expect(isMonotonic(&[_]i64{ 1, 2, 2, 3 }));
    try testing.expect(isMonotonic(&[_]i64{ 6, 5, 4, 4 }));
    try testing.expect(!isMonotonic(&[_]i64{ 1, 3, 2 }));
    try testing.expect(!isMonotonic(&[_]i64{ 1, 2, 3, 4, 5, 6, 5 }));
    try testing.expect(isMonotonic(&[_]i64{ -3, -2, -1 }));
    try testing.expect(isMonotonic(&[_]i64{ -5, -6, -7 }));
    try testing.expect(isMonotonic(&[_]i64{ 0, 0, 0 }));
    try testing.expect(isMonotonic(&[_]i64{ -100, 0, 100 }));
}

test "monotonic array: boundary and extreme" {
    try testing.expect(isMonotonic(&[_]i64{}));
    try testing.expect(isMonotonic(&[_]i64{42}));

    const n: usize = 200_000;
    const ascending = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(ascending);
    for (0..n) |i| ascending[i] = @intCast(i);
    try testing.expect(isMonotonic(ascending));

    const descending = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(descending);
    for (0..n) |i| descending[i] = @intCast(n - i);
    try testing.expect(isMonotonic(descending));

    descending[n / 2] = -9_999_999;
    descending[n / 2 + 1] = 9_999_999;
    try testing.expect(!isMonotonic(descending));
}
