//! Charging Capacitor - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/charging_capacitor.py

const std = @import("std");
const testing = std.testing;

pub const ChargingCapacitorError = error{
    NonPositiveSourceVoltage,
    NonPositiveResistance,
    NonPositiveCapacitance,
};

fn roundTo3(value: f64) f64 {
    return @round(value * 1000.0) / 1000.0;
}

/// Computes capacitor voltage during charging in an RC circuit:
/// Vc = Vs * (1 - exp(-t / (R * C))), rounded to 3 decimals.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn chargingCapacitor(
    source_voltage: f64,
    resistance: f64,
    capacitance: f64,
    time_sec: f64,
) ChargingCapacitorError!f64 {
    if (source_voltage <= 0) return ChargingCapacitorError.NonPositiveSourceVoltage;
    if (resistance <= 0) return ChargingCapacitorError.NonPositiveResistance;
    if (capacitance <= 0) return ChargingCapacitorError.NonPositiveCapacitance;

    const value = source_voltage * (1.0 - std.math.exp(-time_sec / (resistance * capacitance)));
    return roundTo3(value);
}

test "charging capacitor: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.013), try chargingCapacitor(0.2, 0.9, 8.4, 0.5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.446), try chargingCapacitor(2.2, 3.5, 2.4, 9), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.007), try chargingCapacitor(15, 200, 20, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 19.975), try chargingCapacitor(20, 2000, 30e-5, 4), 1e-12);

    try testing.expectError(ChargingCapacitorError.NonPositiveSourceVoltage, chargingCapacitor(0, 10.0, 0.30, 3));
    try testing.expectError(ChargingCapacitorError.NonPositiveResistance, chargingCapacitor(20, -2000, 30, 4));
    try testing.expectError(ChargingCapacitorError.NonPositiveCapacitance, chargingCapacitor(30, 1500, 0, 4));
}

test "charging capacitor: boundary and extreme values" {
    const near_source = try chargingCapacitor(10, 1, 1, 1000);
    try testing.expectApproxEqAbs(@as(f64, 10.0), near_source, 1e-9);

    const negative_time = try chargingCapacitor(5, 2, 2, -1);
    try testing.expect(std.math.isFinite(negative_time));
}
