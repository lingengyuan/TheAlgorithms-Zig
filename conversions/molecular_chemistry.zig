//! Molecular Chemistry Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/molecular_chemistry.py

const std = @import("std");
const testing = std.testing;

pub const MolecularChemistryError = error{DivisionByZero};

fn roundToNearest(value: f64) f64 {
    return std.math.round(value);
}

/// Converts molarity to normality.
/// Formula: round((moles / volume) * nfactor)
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn molarityToNormality(nfactor: i64, moles: f64, volume: f64) MolecularChemistryError!f64 {
    if (volume == 0) return MolecularChemistryError.DivisionByZero;
    return roundToNearest((moles / volume) * @as(f64, @floatFromInt(nfactor)));
}

/// Converts moles to pressure using ideal gas equation.
/// Formula: round((moles * 0.0821 * temperature) / volume)
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn molesToPressure(volume: f64, moles: f64, temperature: f64) MolecularChemistryError!f64 {
    if (volume == 0) return MolecularChemistryError.DivisionByZero;
    return roundToNearest((moles * 0.0821 * temperature) / volume);
}

/// Converts moles to volume using ideal gas equation.
/// Formula: round((moles * 0.0821 * temperature) / pressure)
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn molesToVolume(pressure: f64, moles: f64, temperature: f64) MolecularChemistryError!f64 {
    if (pressure == 0) return MolecularChemistryError.DivisionByZero;
    return roundToNearest((moles * 0.0821 * temperature) / pressure);
}

/// Converts pressure and volume to temperature using ideal gas equation.
/// Formula: round((pressure * volume) / (0.0821 * moles))
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn pressureAndVolumeToTemperature(
    pressure: f64,
    moles: f64,
    volume: f64,
) MolecularChemistryError!f64 {
    if (moles == 0) return MolecularChemistryError.DivisionByZero;
    return roundToNearest((pressure * volume) / (0.0821 * moles));
}

test "molecular chemistry: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 20), try molarityToNormality(2, 3.1, 0.31), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8), try molarityToNormality(4, 11.4, 5.7), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 90), try molesToPressure(0.82, 3, 300), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10), try molesToPressure(8.2, 5, 200), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 90), try molesToVolume(0.82, 3, 300), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10), try molesToVolume(8.2, 5, 200), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 20), try pressureAndVolumeToTemperature(0.82, 1, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 60), try pressureAndVolumeToTemperature(8.2, 5, 3), 1e-12);
}

test "molecular chemistry: division by zero boundaries" {
    try testing.expectError(MolecularChemistryError.DivisionByZero, molarityToNormality(1, 1, 0));
    try testing.expectError(MolecularChemistryError.DivisionByZero, molesToPressure(0, 1, 1));
    try testing.expectError(MolecularChemistryError.DivisionByZero, molesToVolume(0, 1, 1));
    try testing.expectError(MolecularChemistryError.DivisionByZero, pressureAndVolumeToTemperature(1, 0, 1));
}

test "molecular chemistry: extreme magnitudes" {
    const high = try molesToPressure(1e-6, 1e6, 1e3);
    try testing.expect(high > 1e13);

    const tiny = try molesToVolume(1e6, 1e-6, 1e-3);
    try testing.expect(tiny >= 0);
}
