//! Horizontal Projectile Motion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/horizontal_projectile_motion.py

const std = @import("std");
const testing = std.testing;

pub const ProjectileMotionError = error{
    InvalidAngleRange,
    NegativeVelocity,
};

const earth_gravity: f64 = 9.80665;

fn roundToTwoDecimals(value: f64) f64 {
    return @round(value * 100.0) / 100.0;
}

fn validateArgs(init_velocity: f64, angle: f64) ProjectileMotionError!void {
    if (angle < 1 or angle > 90) {
        return ProjectileMotionError.InvalidAngleRange;
    }
    if (init_velocity < 0) {
        return ProjectileMotionError.NegativeVelocity;
    }
}

/// Computes horizontal distance:
/// d = v0^2 * sin(2a) / g.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn horizontalDistance(init_velocity: f64, angle: f64) ProjectileMotionError!f64 {
    try validateArgs(init_velocity, angle);
    const radians = (2.0 * angle) * std.math.pi / 180.0;
    return roundToTwoDecimals(init_velocity * init_velocity * @sin(radians) / earth_gravity);
}

/// Computes maximum height:
/// h = v0^2 * sin^2(a) / (2g).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn maxHeight(init_velocity: f64, angle: f64) ProjectileMotionError!f64 {
    try validateArgs(init_velocity, angle);
    const radians = angle * std.math.pi / 180.0;
    const sine = @sin(radians);
    return roundToTwoDecimals(init_velocity * init_velocity * sine * sine / (2.0 * earth_gravity));
}

/// Computes total flight time:
/// t = 2 * v0 * sin(a) / g.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn totalTime(init_velocity: f64, angle: f64) ProjectileMotionError!f64 {
    try validateArgs(init_velocity, angle);
    const radians = angle * std.math.pi / 180.0;
    return roundToTwoDecimals(2.0 * init_velocity * @sin(radians) / earth_gravity);
}

test "horizontal projectile motion: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 91.77), try horizontalDistance(30, 45), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 414.76), try horizontalDistance(100, 78), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 22.94), try maxHeight(30, 45), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 487.82), try maxHeight(100, 78), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 4.33), try totalTime(30, 45), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 19.95), try totalTime(100, 78), 1e-12);
}

test "horizontal projectile motion: validation and extreme values" {
    try testing.expectError(ProjectileMotionError.NegativeVelocity, horizontalDistance(-1, 20));
    try testing.expectError(ProjectileMotionError.InvalidAngleRange, horizontalDistance(30, -20));
    try testing.expectError(ProjectileMotionError.InvalidAngleRange, maxHeight(30, 0));
    try testing.expectError(ProjectileMotionError.InvalidAngleRange, totalTime(30, 91));

    const high_energy_range = try horizontalDistance(1e6, 45);
    try testing.expect(std.math.isFinite(high_energy_range));
    try testing.expect(high_energy_range > 1e11);

    try testing.expectApproxEqAbs(@as(f64, 0.0), try horizontalDistance(100, 90), 1e-12);
}
