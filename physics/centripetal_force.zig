//! Centripetal Force - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/centripetal_force.py

const std = @import("std");
const testing = std.testing;

pub const CentripetalError = error{
    NegativeMass,
    NonPositiveRadius,
};

/// Computes centripetal force using F = m * v^2 / r.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn centripetal(mass: f64, velocity: f64, radius: f64) CentripetalError!f64 {
    if (mass < 0) return CentripetalError.NegativeMass;
    if (radius <= 0) return CentripetalError.NonPositiveRadius;
    return (mass * velocity * velocity) / radius;
}

test "centripetal force: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1395.0), try centripetal(15.5, -30, 10), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 450.0), try centripetal(10, 15, 5), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3333.3333333333335), try centripetal(20, -50, 15), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 784.0), try centripetal(12.25, 40, 25), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 10000.0), try centripetal(50, 100, 50), 1e-9);
}

test "centripetal force: invalid inputs" {
    try testing.expectError(CentripetalError.NegativeMass, centripetal(-1, 10, 1));
    try testing.expectError(CentripetalError.NonPositiveRadius, centripetal(1, 10, 0));
    try testing.expectError(CentripetalError.NonPositiveRadius, centripetal(1, 10, -1));
}

test "centripetal force: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try centripetal(0, 100, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try centripetal(10, 0, 10), 1e-12);

    const huge = try centripetal(1e150, 1e60, 1e10);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
