//! Reynolds Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/reynolds_number.py

const std = @import("std");
const testing = std.testing;

pub const ReynoldsNumberError = error{
    NonPositiveDensityDiameterOrViscosity,
};

/// Computes Reynolds number:
/// Re = density * |velocity| * diameter / viscosity.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn reynoldsNumber(
    density: f64,
    velocity: f64,
    diameter: f64,
    viscosity: f64,
) ReynoldsNumberError!f64 {
    if (density <= 0 or diameter <= 0 or viscosity <= 0) {
        return ReynoldsNumberError.NonPositiveDensityDiameterOrViscosity;
    }
    return (density * @abs(velocity) * diameter) / viscosity;
}

test "reynolds number: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 281.25), try reynoldsNumber(900, 2.5, 0.05, 0.4), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 589.0695652173912), try reynoldsNumber(450, 3.86, 0.078, 0.23), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 717.9545454545454), try reynoldsNumber(234, -4.5, 0.3, 0.44), 1e-12);

    try testing.expectError(ReynoldsNumberError.NonPositiveDensityDiameterOrViscosity, reynoldsNumber(-90, 2, 0.045, 1));
    try testing.expectError(ReynoldsNumberError.NonPositiveDensityDiameterOrViscosity, reynoldsNumber(0, 2, -0.4, -2));
}

test "reynolds number: boundary and extreme values" {
    const tiny = try reynoldsNumber(1e-12, 1e-9, 1e-12, 1e-18);
    try testing.expect(std.math.isFinite(tiny));
    try testing.expect(tiny > 0);

    const huge = try reynoldsNumber(1e20, 1e20, 1e10, 1e-10);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
