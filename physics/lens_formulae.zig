//! Lens Formulae - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/lens_formulae.py

const std = @import("std");
const testing = std.testing;

pub const LensFormulaError = error{
    InvalidInputZero,
};

/// Computes focal length from object and image distances:
/// f = 1 / ((1 / v) - (1 / u)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn focalLengthOfLens(
    object_distance_from_lens: f64,
    image_distance_from_lens: f64,
) LensFormulaError!f64 {
    if (object_distance_from_lens == 0 or image_distance_from_lens == 0) {
        return LensFormulaError.InvalidInputZero;
    }
    return 1.0 / ((1.0 / image_distance_from_lens) - (1.0 / object_distance_from_lens));
}

/// Computes object distance from focal length and image distance:
/// u = 1 / ((1 / v) - (1 / f)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn objectDistance(
    focal_length_of_lens: f64,
    image_distance_from_lens: f64,
) LensFormulaError!f64 {
    if (focal_length_of_lens == 0 or image_distance_from_lens == 0) {
        return LensFormulaError.InvalidInputZero;
    }
    return 1.0 / ((1.0 / image_distance_from_lens) - (1.0 / focal_length_of_lens));
}

/// Computes image distance from focal length and object distance:
/// v = 1 / ((1 / u) + (1 / f)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn imageDistance(
    focal_length_of_lens: f64,
    object_distance_from_lens: f64,
) LensFormulaError!f64 {
    if (focal_length_of_lens == 0 or object_distance_from_lens == 0) {
        return LensFormulaError.InvalidInputZero;
    }
    return 1.0 / ((1.0 / object_distance_from_lens) + (1.0 / focal_length_of_lens));
}

test "lens formulae: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 6.666666666666667), try focalLengthOfLens(10, 4), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -5.0516129032258075), try focalLengthOfLens(2.7, 5.8), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, -13.333333333333332), try objectDistance(10, 40), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.9787234042553192), try objectDistance(6.2, 1.5), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 22.22222222222222), try imageDistance(50, 40), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.1719696969696973), try imageDistance(5.3, 7.9), 1e-12);
}

test "lens formulae: validation and extreme values" {
    try testing.expectError(LensFormulaError.InvalidInputZero, focalLengthOfLens(0, 20));
    try testing.expectError(LensFormulaError.InvalidInputZero, objectDistance(0, 20));
    try testing.expectError(LensFormulaError.InvalidInputZero, imageDistance(0, 20));

    const extreme = try imageDistance(1e-12, 1e12);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
