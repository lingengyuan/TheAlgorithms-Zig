//! Newton's Second Law of Motion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/newtons_second_law_of_motion.py

const std = @import("std");
const testing = std.testing;

/// Computes force using Newton's second law: F = m * a.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn newtonsSecondLawOfMotion(mass: f64, acceleration: f64) f64 {
    return mass * acceleration;
}

test "newtons second law: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 100.0), newtonsSecondLawOfMotion(10, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), newtonsSecondLawOfMotion(2.0, 1), 1e-12);
}

test "newtons second law: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), newtonsSecondLawOfMotion(0.0, 123.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -50.0), newtonsSecondLawOfMotion(10.0, -5.0), 1e-12);

    const huge = newtonsSecondLawOfMotion(1e150, 1e90);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
