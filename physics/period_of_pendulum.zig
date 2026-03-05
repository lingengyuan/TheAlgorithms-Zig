//! Period of Pendulum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/period_of_pendulum.py

const std = @import("std");
const testing = std.testing;

pub const PeriodOfPendulumError = error{
    NegativeLength,
};

const gravity: f64 = 9.80665;

/// Computes the period of a simple pendulum:
/// T = 2 * pi * sqrt(length / g).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn periodOfPendulum(length: f64) PeriodOfPendulumError!f64 {
    if (length < 0) return PeriodOfPendulumError.NegativeLength;
    return 2.0 * std.math.pi * @sqrt(length / gravity);
}

test "period of pendulum: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.2252155506257845), try periodOfPendulum(1.23), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.0888278441908574), try periodOfPendulum(2.37), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.76073193364765), try periodOfPendulum(5.63), 1e-12);
    try testing.expectError(PeriodOfPendulumError.NegativeLength, periodOfPendulum(-12));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try periodOfPendulum(0), 1e-12);
}

test "period of pendulum: boundary and extreme values" {
    const tiny = try periodOfPendulum(1e-24);
    try testing.expect(std.math.isFinite(tiny));
    try testing.expect(tiny >= 0);

    const huge = try periodOfPendulum(1e30);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
