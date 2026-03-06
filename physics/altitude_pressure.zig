//! Altitude from Pressure - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/altitude_pressure.py

const std = @import("std");
const testing = std.testing;

pub const AltitudePressureError = error{
    ValueHigherThanSeaLevelPressure,
    NegativeAtmosphericPressure,
};

const sea_level_pressure: f64 = 101_325.0;
const altitude_scale: f64 = 44_330.0;

/// Approximates altitude from atmospheric pressure:
/// H = 44330 * (1 - (P / P0)^(1 / 5.5255)).
///
/// Note: Exponent follows the Python reference implementation.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn getAltitudeAtPressure(pressure: f64) AltitudePressureError!f64 {
    if (pressure > sea_level_pressure) {
        return AltitudePressureError.ValueHigherThanSeaLevelPressure;
    }
    if (pressure < 0) {
        return AltitudePressureError.NegativeAtmosphericPressure;
    }

    return altitude_scale * (1.0 - std.math.pow(f64, pressure / sea_level_pressure, 1.0 / 5.5255));
}

test "altitude pressure: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 105.47836610778828), try getAltitudeAtPressure(100_000), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try getAltitudeAtPressure(101_325), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1855.873388064995), try getAltitudeAtPressure(80_000), 1e-9);
}

test "altitude pressure: validation and extreme values" {
    try testing.expectError(AltitudePressureError.ValueHigherThanSeaLevelPressure, getAltitudeAtPressure(201_325));
    try testing.expectError(AltitudePressureError.NegativeAtmosphericPressure, getAltitudeAtPressure(-80_000));

    const vacuum = try getAltitudeAtPressure(0);
    try testing.expect(std.math.isFinite(vacuum));
    try testing.expect(vacuum > 40_000);

    const almost_sea_level = try getAltitudeAtPressure(101_324.999999);
    try testing.expect(almost_sea_level >= 0);
    try testing.expect(almost_sea_level < 1e-3);
}
