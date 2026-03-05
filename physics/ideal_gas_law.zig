//! Ideal Gas Law - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/ideal_gas_law.py

const std = @import("std");
const testing = std.testing;

pub const IdealGasLawError = error{
    InvalidInputs,
    DivisionByZero,
};

pub const universalGasConstant: f64 = 8.314462;

/// Computes pressure from moles, temperature, and volume:
/// P = nRT / V.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn pressureOfGasSystem(moles: f64, kelvin: f64, volume: f64) IdealGasLawError!f64 {
    if (moles < 0 or kelvin < 0 or volume < 0) return IdealGasLawError.InvalidInputs;
    if (volume == 0) return IdealGasLawError.DivisionByZero;
    return moles * kelvin * universalGasConstant / volume;
}

/// Computes volume from moles, temperature, and pressure:
/// V = nRT / P.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn volumeOfGasSystem(moles: f64, kelvin: f64, pressure: f64) IdealGasLawError!f64 {
    if (moles < 0 or kelvin < 0 or pressure < 0) return IdealGasLawError.InvalidInputs;
    if (pressure == 0) return IdealGasLawError.DivisionByZero;
    return moles * kelvin * universalGasConstant / pressure;
}

/// Computes temperature from moles, volume, and pressure:
/// T = PV / (nR).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn temperatureOfGasSystem(moles: f64, volume: f64, pressure: f64) IdealGasLawError!f64 {
    if (moles < 0 or volume < 0 or pressure < 0) return IdealGasLawError.InvalidInputs;
    if (moles == 0) return IdealGasLawError.DivisionByZero;
    return pressure * volume / (moles * universalGasConstant);
}

/// Computes moles from temperature, volume, and pressure:
/// n = PV / (TR).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn molesOfGasSystem(kelvin: f64, volume: f64, pressure: f64) IdealGasLawError!f64 {
    if (kelvin < 0 or volume < 0 or pressure < 0) return IdealGasLawError.InvalidInputs;
    if (kelvin == 0) return IdealGasLawError.DivisionByZero;
    return pressure * volume / (kelvin * universalGasConstant);
}

test "ideal gas law: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 332.57848), try pressureOfGasSystem(2, 100, 5), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 283731.01575), try pressureOfGasSystem(0.5, 273, 0.004), 1e-6);
    try testing.expectError(IdealGasLawError.InvalidInputs, pressureOfGasSystem(3, -0.46, 23.5));

    try testing.expectApproxEqAbs(@as(f64, 332.57848), try volumeOfGasSystem(2, 100, 5), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 283731.01575), try volumeOfGasSystem(0.5, 273, 0.004), 1e-6);
    try testing.expectError(IdealGasLawError.InvalidInputs, volumeOfGasSystem(3, -0.46, 23.5));

    try testing.expectApproxEqAbs(@as(f64, 30.068090996146232), try temperatureOfGasSystem(2, 100, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 54767.66101807144), try temperatureOfGasSystem(11, 5009, 1000), 1e-9);
    try testing.expectError(IdealGasLawError.InvalidInputs, temperatureOfGasSystem(3, -0.46, 23.5));

    try testing.expectApproxEqAbs(@as(f64, 0.06013618199229246), try molesOfGasSystem(100, 5, 10), 1e-18);
    try testing.expectApproxEqAbs(@as(f64, 5476.766101807144), try molesOfGasSystem(110, 5009, 1000), 1e-9);
    try testing.expectError(IdealGasLawError.InvalidInputs, molesOfGasSystem(3, -0.46, 23.5));
}

test "ideal gas law: boundary and extreme values" {
    try testing.expectError(IdealGasLawError.DivisionByZero, pressureOfGasSystem(1, 1, 0));
    try testing.expectError(IdealGasLawError.DivisionByZero, volumeOfGasSystem(1, 1, 0));
    try testing.expectError(IdealGasLawError.DivisionByZero, temperatureOfGasSystem(0, 1, 1));
    try testing.expectError(IdealGasLawError.DivisionByZero, molesOfGasSystem(0, 1, 1));

    const huge_p = try pressureOfGasSystem(1e20, 1e5, 1e-2);
    try testing.expect(std.math.isFinite(huge_p));
    try testing.expect(huge_p > 0);
}
