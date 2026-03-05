//! RMS Speed of Molecule - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/rms_speed_of_molecule.py

const std = @import("std");
const testing = std.testing;

pub const RmsSpeedError = error{
    NegativeTemperature,
    NonPositiveMolarMass,
};

pub const universalGasConstant: f64 = 8.3144598;

/// Computes root-mean-square molecular speed:
/// Vrms = sqrt(3 * R * T / M).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn rmsSpeedOfMolecule(temperature: f64, molar_mass: f64) RmsSpeedError!f64 {
    if (temperature < 0) return RmsSpeedError.NegativeTemperature;
    if (molar_mass <= 0) return RmsSpeedError.NonPositiveMolarMass;
    return @sqrt(3.0 * universalGasConstant * temperature / molar_mass);
}

test "rms speed of molecule: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 35.315279554323226), try rmsSpeedOfMolecule(100, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 23.821458421977443), try rmsSpeedOfMolecule(273, 12), 1e-12);
}

test "rms speed of molecule: invalid inputs" {
    try testing.expectError(RmsSpeedError.NegativeTemperature, rmsSpeedOfMolecule(-1, 2));
    try testing.expectError(RmsSpeedError.NonPositiveMolarMass, rmsSpeedOfMolecule(100, 0));
    try testing.expectError(RmsSpeedError.NonPositiveMolarMass, rmsSpeedOfMolecule(100, -1));
}

test "rms speed of molecule: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try rmsSpeedOfMolecule(0, 2), 1e-12);

    const huge = try rmsSpeedOfMolecule(1e9, 1e-9);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
