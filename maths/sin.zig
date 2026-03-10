//! Sin - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sin.py

const std = @import("std");
const testing = std.testing;
const maclaurin = @import("maclaurin_series.zig");

pub const SinError = error{InvalidAccuracy};

/// Approximates `sin(angle_in_degrees)` using the Maclaurin series.
/// Input and rounding semantics follow the Python reference.
/// Time complexity: O(accuracy), Space complexity: O(1)
pub fn sin(
    angle_in_degrees: f64,
    accuracy: i64,
    rounded_values_count: u8,
) SinError!f64 {
    if (accuracy <= 0) return error.InvalidAccuracy;
    const normalized_degrees = angle_in_degrees - @floor(angle_in_degrees / 360.0) * 360.0;
    const radians = normalized_degrees * std.math.pi / 180.0;
    const raw = try maclaurin.maclaurinSin(radians, accuracy);
    return roundTo(raw, rounded_values_count);
}

fn roundTo(value: f64, decimals: u8) f64 {
    var scale: f64 = 1.0;
    for (0..decimals) |_| scale *= 10.0;
    return @round(value * scale) / scale;
}

test "sin: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try sin(0.0, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try sin(90.0, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try sin(180.0, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -1.0), try sin(270.0, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0118679603), try sin(0.68, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0343762121), try sin(1.97, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.8987940463), try sin(64.0, 18, 10), 1e-10);
}

test "sin: normalization and extreme examples" {
    try testing.expectApproxEqAbs(@as(f64, -0.9876883406), try sin(9999.0, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.5150380749), try sin(-689.0, 18, 10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.9999862922), try sin(89.7, 18, 10), 1e-10);
    try testing.expectError(error.InvalidAccuracy, sin(30.0, 0, 10));
}
