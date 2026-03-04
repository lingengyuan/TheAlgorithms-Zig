//! Geometric Mean - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/geometric_mean.py

const std = @import("std");
const testing = std.testing;

pub const GeometricMeanError = error{ InvalidInput, NotANumber, CannotCompute };

/// Returns geometric mean of input numbers.
/// Time complexity: O(n), Space complexity: O(1)
pub fn computeGeometricMean(args: []const f64) GeometricMeanError!f64 {
    if (args.len == 0) return GeometricMeanError.InvalidInput;

    var product: f64 = 1.0;
    for (args) |number| {
        if (!std.math.isFinite(number)) return GeometricMeanError.NotANumber;
        product *= number;
    }

    if (product < 0 and args.len % 2 == 0) return GeometricMeanError.CannotCompute;

    var mean = std.math.pow(f64, @abs(product), 1.0 / @as(f64, @floatFromInt(args.len)));
    if (product < 0) mean = -mean;

    const rounded = std.math.round(mean);
    const back = std.math.pow(f64, rounded, @as(f64, @floatFromInt(args.len)));
    if (back == product) mean = rounded;

    return mean;
}

test "geometric mean: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 4.0), try computeGeometricMean(&[_]f64{ 2, 8 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 25.0), try computeGeometricMean(&[_]f64{ 5, 125 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try computeGeometricMean(&[_]f64{ 1, 0 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try computeGeometricMean(&[_]f64{ 1, 5, 25, 5 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -5.0), try computeGeometricMean(&[_]f64{ -5, 25, 1 }), 1e-12);
}

test "geometric mean: invalid and extreme cases" {
    try testing.expectError(GeometricMeanError.InvalidInput, computeGeometricMean(&[_]f64{}));
    try testing.expectError(GeometricMeanError.CannotCompute, computeGeometricMean(&[_]f64{ 2, -2 }));
    try testing.expectError(GeometricMeanError.NotANumber, computeGeometricMean(&[_]f64{ std.math.inf(f64), 4 }));

    const many = [_]f64{1.0001} ** 2_000;
    const gm = try computeGeometricMean(&many);
    try testing.expectApproxEqAbs(@as(f64, 1.0001), gm, 1e-8);
}
