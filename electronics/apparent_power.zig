//! Apparent Power - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/apparent_power.py

const std = @import("std");
const testing = std.testing;

pub const ComplexValue = struct {
    real: f64,
    imag: f64,
};

fn degreesToRadians(degrees: f64) f64 {
    return degrees * std.math.pi / 180.0;
}

/// Computes apparent power (complex) in single-phase AC circuit by multiplying
/// voltage and current phasors in rectangular form.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn apparentPower(
    voltage: f64,
    current: f64,
    voltage_angle: f64,
    current_angle: f64,
) ComplexValue {
    const magnitude = voltage * current;
    const theta = degreesToRadians(voltage_angle + current_angle);
    return ComplexValue{
        .real = magnitude * @cos(theta),
        .imag = magnitude * @sin(theta),
    };
}

test "apparent power: python examples" {
    const p1 = apparentPower(100, 5, 0, 0);
    try testing.expectApproxEqAbs(@as(f64, 500.0), p1.real, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), p1.imag, 1e-12);

    const p2 = apparentPower(100, 5, 90, 0);
    try testing.expectApproxEqAbs(@as(f64, 3.061616997868383e-14), p2.real, 1e-20);
    try testing.expectApproxEqAbs(@as(f64, 500.0), p2.imag, 1e-12);

    const p3 = apparentPower(100, 5, -45, -60);
    try testing.expectApproxEqAbs(@as(f64, -129.40952255126027), p3.real, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -482.9629131445341), p3.imag, 1e-12);

    const p4 = apparentPower(200, 10, -30, -90);
    try testing.expectApproxEqAbs(@as(f64, -999.9999999999998), p4.real, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -1732.0508075688776), p4.imag, 1e-9);
}

test "apparent power: boundary and extreme values" {
    const zero = apparentPower(0, 10, 0, 45);
    try testing.expectApproxEqAbs(@as(f64, 0.0), zero.real, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), zero.imag, 1e-12);

    const huge = apparentPower(1e150, 1e150, 30, -30);
    try testing.expect(std.math.isFinite(huge.real));
    try testing.expectApproxEqAbs(@as(f64, 0.0), huge.imag, 1e285);
}
