//! Mirror Formulae - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/mirror_formulae.py

const std = @import("std");
const testing = std.testing;

pub const MirrorFormulaError = error{
    InvalidInputZero,
};

/// Computes mirror focal length:
/// f = 1 / ((1 / u) + (1 / v)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn focalLength(distance_of_object: f64, distance_of_image: f64) MirrorFormulaError!f64 {
    if (distance_of_object == 0 or distance_of_image == 0) {
        return MirrorFormulaError.InvalidInputZero;
    }
    return 1.0 / ((1.0 / distance_of_object) + (1.0 / distance_of_image));
}

/// Computes object distance:
/// u = 1 / ((1 / f) - (1 / v)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn objectDistance(focal_length: f64, distance_of_image: f64) MirrorFormulaError!f64 {
    if (distance_of_image == 0 or focal_length == 0) {
        return MirrorFormulaError.InvalidInputZero;
    }
    return 1.0 / ((1.0 / focal_length) - (1.0 / distance_of_image));
}

/// Computes image distance:
/// v = 1 / ((1 / f) - (1 / u)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn imageDistance(focal_length: f64, distance_of_object: f64) MirrorFormulaError!f64 {
    if (distance_of_object == 0 or focal_length == 0) {
        return MirrorFormulaError.InvalidInputZero;
    }
    return 1.0 / ((1.0 / focal_length) - (1.0 / distance_of_object));
}

test "mirror formulae: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 6.66666666666666), try focalLength(10, 20), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.929012346), try focalLength(9.5, 6.7), 1e-9);

    try testing.expectApproxEqAbs(@as(f64, -60.0), try objectDistance(30, 20), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 102.375), try objectDistance(10.5, 11.7), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 13.33333333), try imageDistance(10, 40), 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 1.932692308), try imageDistance(1.5, 6.7), 1e-9);
}

test "mirror formulae: validation and extreme values" {
    try testing.expectError(MirrorFormulaError.InvalidInputZero, focalLength(0, 20));
    try testing.expectError(MirrorFormulaError.InvalidInputZero, objectDistance(90, 0));
    try testing.expectError(MirrorFormulaError.InvalidInputZero, imageDistance(0, 0));

    const extreme = try imageDistance(1e-6, 1e6);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
