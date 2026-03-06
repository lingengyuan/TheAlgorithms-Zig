//! Basic Orbital Capture - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/basic_orbital_capture.py

const std = @import("std");
const testing = std.testing;

pub const BasicOrbitalCaptureError = error{
    RadiusMustBePositive,
    MassCannotBeNegative,
    ProjectileVelocityMustBePositive,
    BeyondSpeedOfLight,
    CaptureRadiusCannotBeNegative,
};

const gravitational_constant: f64 = 6.67430e-11;
const speed_of_light: f64 = 299_792_458.0;

/// Computes gravitational capture radius:
/// R_capture = R_target * sqrt(1 + v_escape^2 / v_projectile^2),
/// where v_escape^2 = 2GM / R.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn captureRadii(
    target_body_radius: f64,
    target_body_mass: f64,
    projectile_velocity: f64,
) BasicOrbitalCaptureError!f64 {
    if (target_body_mass < 0) {
        return BasicOrbitalCaptureError.MassCannotBeNegative;
    }
    if (target_body_radius <= 0) {
        return BasicOrbitalCaptureError.RadiusMustBePositive;
    }
    if (projectile_velocity <= 0) {
        return BasicOrbitalCaptureError.ProjectileVelocityMustBePositive;
    }
    if (projectile_velocity > speed_of_light) {
        return BasicOrbitalCaptureError.BeyondSpeedOfLight;
    }

    const escape_velocity_squared = (2.0 * gravitational_constant * target_body_mass) / target_body_radius;
    const capture_radius = target_body_radius * @sqrt(1.0 + escape_velocity_squared / (projectile_velocity * projectile_velocity));
    return @round(capture_radius);
}

/// Computes effective capture area:
/// sigma = pi * capture_radius^2.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn captureArea(capture_radius: f64) BasicOrbitalCaptureError!f64 {
    if (capture_radius < 0) {
        return BasicOrbitalCaptureError.CaptureRadiusCannotBeNegative;
    }

    const sigma = std.math.pi * capture_radius * capture_radius;
    return @round(sigma);
}

test "basic orbital capture: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 17_209_590_691.0), try captureRadii(6.957e8, 1.99e30, 25_000.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 9.304455331329126e20), try captureArea(17_209_590_691.0), 1e6);
}

test "basic orbital capture: validation and extreme values" {
    try testing.expectError(BasicOrbitalCaptureError.RadiusMustBePositive, captureRadii(-6.957e8, 1.99e30, 25_000.0));
    try testing.expectError(BasicOrbitalCaptureError.MassCannotBeNegative, captureRadii(6.957e8, -1.99e30, 25_000.0));
    try testing.expectError(BasicOrbitalCaptureError.BeyondSpeedOfLight, captureRadii(6.957e8, 1.99e30, speed_of_light + 1.0));
    try testing.expectError(BasicOrbitalCaptureError.CaptureRadiusCannotBeNegative, captureArea(-1));

    const huge_capture = try captureRadii(6.371e6, 5.972e24, 1.0);
    try testing.expect(std.math.isFinite(huge_capture));
    try testing.expect(huge_capture > 1e10);

    const area = try captureArea(huge_capture);
    try testing.expect(std.math.isFinite(area));
    try testing.expect(area > 0);
}
