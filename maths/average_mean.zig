//! Average Mean - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/average_mean.py

const std = @import("std");
const testing = std.testing;

pub const MeanError = error{EmptyInput};

/// Returns arithmetic mean of numbers in `nums`.
/// Time complexity: O(n), Space complexity: O(1)
pub fn mean(nums: []const f64) MeanError!f64 {
    if (nums.len == 0) return MeanError.EmptyInput;

    var total: f64 = 0.0;
    for (nums) |value| total += value;
    return total / @as(f64, @floatFromInt(nums.len));
}

test "average mean: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 12.0), try mean(&[_]f64{ 3, 6, 9, 12, 15, 18, 21 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 20.0), try mean(&[_]f64{ 5, 10, 15, 20, 25, 30, 35 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.5), try mean(&[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }), 1e-12);
}

test "average mean: edge and extreme cases" {
    try testing.expectError(MeanError.EmptyInput, mean(&[_]f64{}));

    var many = [_]f64{0} ** 10_000;
    @memset(&many, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try mean(&many), 1e-12);
}
