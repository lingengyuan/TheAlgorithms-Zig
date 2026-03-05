//! Rectangular to Polar Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/rectangular_to_polar.py

const std = @import("std");
const testing = std.testing;

pub const Polar = struct {
    magnitude: f64,
    angle_degrees: f64,
};

fn round2(value: f64) f64 {
    return std.math.round(value * 100.0) / 100.0;
}

/// Converts rectangular coordinates `(real, img)` to polar `(magnitude, angle_degrees)`,
/// rounded to 2 decimals.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn rectangularToPolar(real: f64, img: f64) Polar {
    const magnitude = round2(std.math.sqrt(real * real + img * img));
    const angle = round2(std.math.radiansToDegrees(std.math.atan2(img, real)));
    return .{ .magnitude = magnitude, .angle_degrees = angle };
}

test "rectangular to polar: python examples" {
    const p1 = rectangularToPolar(5, -5);
    try testing.expectApproxEqAbs(@as(f64, 7.07), p1.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -45.0), p1.angle_degrees, 1e-12);

    const p2 = rectangularToPolar(-1, 1);
    try testing.expectApproxEqAbs(@as(f64, 1.41), p2.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 135.0), p2.angle_degrees, 1e-12);

    const p3 = rectangularToPolar(-1, -1);
    try testing.expectApproxEqAbs(@as(f64, 1.41), p3.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -135.0), p3.angle_degrees, 1e-12);

    const p4 = rectangularToPolar(1e-10, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), p4.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 45.0), p4.angle_degrees, 1e-12);

    const p5 = rectangularToPolar(-1e-10, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0.0), p5.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 135.0), p5.angle_degrees, 1e-12);

    const p6 = rectangularToPolar(9.75, 5.93);
    try testing.expectApproxEqAbs(@as(f64, 11.41), p6.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 31.31), p6.angle_degrees, 1e-12);

    const p7 = rectangularToPolar(10000, 99999);
    try testing.expectApproxEqAbs(@as(f64, 100497.76), p7.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 84.29), p7.angle_degrees, 1e-12);
}

test "rectangular to polar: extreme axes" {
    const p1 = rectangularToPolar(0, 0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), p1.magnitude, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), p1.angle_degrees, 1e-12);

    const p2 = rectangularToPolar(0, 1e9);
    try testing.expectApproxEqAbs(@as(f64, 1e9), p2.magnitude, 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 90.0), p2.angle_degrees, 1e-12);
}
