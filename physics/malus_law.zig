//! Malus Law - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/malus_law.py

const std = @import("std");
const testing = std.testing;

pub const MalusLawError = error{
    NegativeIntensity,
    AngleOutOfRange,
};

/// Computes transmitted intensity through a polarizer:
/// I = I0 * cos^2(theta).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn malusLaw(initial_intensity: f64, angle_degrees: f64) MalusLawError!f64 {
    if (initial_intensity < 0) {
        return MalusLawError.NegativeIntensity;
    }
    if (angle_degrees < 0 or angle_degrees > 360) {
        return MalusLawError.AngleOutOfRange;
    }

    const radians = angle_degrees * std.math.pi / 180.0;
    const cosine = @cos(radians);
    return initial_intensity * cosine * cosine;
}

test "malus law: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 5.0), try malusLaw(10, 45), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 25.0), try malusLaw(100, 60), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 37.5), try malusLaw(50, 150), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try malusLaw(75, 270), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 100.0), try malusLaw(100, 180), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 100.0), try malusLaw(100, 360), 1e-12);
}

test "malus law: validation and extreme values" {
    try testing.expectError(MalusLawError.AngleOutOfRange, malusLaw(10, -900));
    try testing.expectError(MalusLawError.AngleOutOfRange, malusLaw(10, 900));
    try testing.expectError(MalusLawError.NegativeIntensity, malusLaw(-100, 180));

    const extreme = try malusLaw(1e300, 89.999);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme >= 0);
}
