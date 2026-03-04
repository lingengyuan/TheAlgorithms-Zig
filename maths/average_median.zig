//! Average Median - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/average_median.py

const std = @import("std");
const testing = std.testing;

pub const MedianError = error{EmptyInput};

/// Returns median of `nums`.
/// Caller allocator is used for sorting copy.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn median(
    allocator: std.mem.Allocator,
    nums: []const f64,
) (MedianError || std.mem.Allocator.Error)!f64 {
    if (nums.len == 0) return MedianError.EmptyInput;

    const sorted = try allocator.alloc(f64, nums.len);
    defer allocator.free(sorted);
    @memcpy(sorted, nums);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));

    const mid = sorted.len >> 1;
    if (sorted.len % 2 == 0) {
        return (sorted[mid] + sorted[mid - 1]) / 2.0;
    }
    return sorted[mid];
}

test "average median: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try median(testing.allocator, &[_]f64{0}), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.5), try median(testing.allocator, &[_]f64{ 4, 1, 3, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), try median(testing.allocator, &[_]f64{ 2, 70, 6, 50, 20, 8, 4 }), 1e-12);
}

test "average median: edge and extreme cases" {
    try testing.expectError(MedianError.EmptyInput, median(testing.allocator, &[_]f64{}));

    const ascending = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try testing.expectApproxEqAbs(@as(f64, 4.5), try median(testing.allocator, &ascending), 1e-12);

    const repeated = [_]f64{ 3, 3, 3, 3, 3, 3 };
    try testing.expectApproxEqAbs(@as(f64, 3.0), try median(testing.allocator, &repeated), 1e-12);
}
