//! Charging Inductor - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/charging_inductor.py

const std = @import("std");
const testing = std.testing;

pub const ChargingInductorError = error{
    NonPositiveSourceVoltage,
    NonPositiveResistance,
    NonPositiveInductance,
};

fn roundTo3(value: f64) f64 {
    return @round(value * 1000.0) / 1000.0;
}

/// Computes inductor current during charging in an RL circuit:
/// I(t) = Vs/R * (1 - exp((-t * R) / L)), rounded to 3 decimals.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn chargingInductor(
    source_voltage: f64,
    resistance: f64,
    inductance: f64,
    time: f64,
) ChargingInductorError!f64 {
    if (source_voltage <= 0) return ChargingInductorError.NonPositiveSourceVoltage;
    if (resistance <= 0) return ChargingInductorError.NonPositiveResistance;
    if (inductance <= 0) return ChargingInductorError.NonPositiveInductance;

    const current = source_voltage / resistance * (1.0 - std.math.exp((-time * resistance) / inductance));
    return roundTo3(current);
}

test "charging inductor: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.817), try chargingInductor(5.8, 1.5, 2.3, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.543), try chargingInductor(8, 5, 3, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.016), try chargingInductor(8, 5e2, 3, 2), 1e-12);

    try testing.expectError(ChargingInductorError.NonPositiveSourceVoltage, chargingInductor(-8, 100, 15, 12));
    try testing.expectError(ChargingInductorError.NonPositiveResistance, chargingInductor(80, -15, 100, 5));
    try testing.expectError(ChargingInductorError.NonPositiveInductance, chargingInductor(12, 200, -20, 5));
    try testing.expectError(ChargingInductorError.NonPositiveSourceVoltage, chargingInductor(0, 200, 20, 5));
    try testing.expectError(ChargingInductorError.NonPositiveResistance, chargingInductor(10, 0, 20, 5));
    try testing.expectError(ChargingInductorError.NonPositiveInductance, chargingInductor(15, 25, 0, 5));
}

test "charging inductor: boundary and extreme values" {
    const near_asymptote = try chargingInductor(10, 5, 2, 1000);
    try testing.expectApproxEqAbs(@as(f64, 2.0), near_asymptote, 1e-9);

    const negative_time = try chargingInductor(10, 5, 2, -1);
    try testing.expect(std.math.isFinite(negative_time));
}
