//! Real and Reactive Power - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/real_and_reactive_power.py

const std = @import("std");
const testing = std.testing;

pub const PowerFactorError = error{
    InvalidPowerFactor,
};

fn validatePowerFactor(power_factor: f64) PowerFactorError!void {
    if (power_factor < -1.0 or power_factor > 1.0) {
        return PowerFactorError.InvalidPowerFactor;
    }
}

/// Computes real power:
/// P = S * power_factor.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn realPower(apparent_power: f64, power_factor: f64) PowerFactorError!f64 {
    try validatePowerFactor(power_factor);
    return apparent_power * power_factor;
}

/// Computes reactive power magnitude:
/// Q = S * sqrt(1 - power_factor^2).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn reactivePower(apparent_power: f64, power_factor: f64) PowerFactorError!f64 {
    try validatePowerFactor(power_factor);
    return apparent_power * @sqrt(1.0 - power_factor * power_factor);
}

test "real and reactive power: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 90.0), try realPower(100, 0.9), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try realPower(0, 0.8), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -90.0), try realPower(100, -0.9), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 43.58898943540673), try reactivePower(100, 0.9), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try reactivePower(0, 0.8), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 43.58898943540673), try reactivePower(100, -0.9), 1e-12);
}

test "real and reactive power: invalid and boundary cases" {
    try testing.expectError(PowerFactorError.InvalidPowerFactor, realPower(100, 1.1));
    try testing.expectError(PowerFactorError.InvalidPowerFactor, reactivePower(100, -1.1));

    try testing.expectApproxEqAbs(@as(f64, 100.0), try realPower(100, 1.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try reactivePower(100, 1.0), 1e-12);
}
