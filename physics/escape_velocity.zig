//! Escape Velocity - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/escape_velocity.py

const std = @import("std");
const testing = std.testing;

pub const EscapeVelocityError = error{
    RadiusCannotBeZero,
    NegativeRadicand,
};

const gravitationalConstant: f64 = 6.67430e-11;

fn roundTo3(value: f64) f64 {
    return @round(value * 1000.0) / 1000.0;
}

/// Computes escape velocity with formula v = sqrt(2 * G * M / R),
/// rounded to 3 decimals, following Python reference behavior.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn escapeVelocity(mass: f64, radius: f64) EscapeVelocityError!f64 {
    if (radius == 0) return EscapeVelocityError.RadiusCannotBeZero;

    const radicand = 2.0 * gravitationalConstant * mass / radius;
    if (radicand < 0) return EscapeVelocityError.NegativeRadicand;

    return roundTo3(@sqrt(radicand));
}

test "escape velocity: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 11185.978), try escapeVelocity(5.972e24, 6.371e6), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 2376.307), try escapeVelocity(7.348e22, 1.737e6), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 60199.545), try escapeVelocity(1.898e27, 6.9911e7), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try escapeVelocity(0, 1.0), 1e-12);
    try testing.expectError(EscapeVelocityError.RadiusCannotBeZero, escapeVelocity(1.0, 0.0));
}

test "escape velocity: boundary and extreme values" {
    try testing.expectError(EscapeVelocityError.NegativeRadicand, escapeVelocity(1.0, -1.0));
    try testing.expectError(EscapeVelocityError.NegativeRadicand, escapeVelocity(-1.0, 1.0));

    const huge = try escapeVelocity(1e35, 10.0);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
