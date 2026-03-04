//! Average Absolute Deviation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/average_absolute_deviation.py

const std = @import("std");
const testing = std.testing;

pub const AADError = error{EmptyInput};

/// Returns average absolute deviation of numbers.
/// Time complexity: O(n), Space complexity: O(1)
pub fn averageAbsoluteDeviation(nums: []const f64) AADError!f64 {
    if (nums.len == 0) return AADError.EmptyInput;

    var sum: f64 = 0.0;
    for (nums) |value| sum += value;
    const avg = sum / @as(f64, @floatFromInt(nums.len));

    var deviation_sum: f64 = 0.0;
    for (nums) |value| deviation_sum += @abs(value - avg);

    return deviation_sum / @as(f64, @floatFromInt(nums.len));
}

test "average absolute deviation: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try averageAbsoluteDeviation(&[_]f64{0}), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try averageAbsoluteDeviation(&[_]f64{ 4, 1, 3, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 20.0), try averageAbsoluteDeviation(&[_]f64{ 2, 70, 6, 50, 20, 8, 4, 0 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 16.25), try averageAbsoluteDeviation(&[_]f64{ -20, 0, 30, 15 }), 1e-12);
}

test "average absolute deviation: edge and extreme cases" {
    try testing.expectError(AADError.EmptyInput, averageAbsoluteDeviation(&[_]f64{}));

    var large = [_]f64{0} ** 100_000;
    @memset(&large, 42.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try averageAbsoluteDeviation(&large), 1e-12);
}
