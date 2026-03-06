//! Speeds of Gas Molecules - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/speeds_of_gas_molecules.py

const std = @import("std");
const testing = std.testing;

pub const GasSpeedError = error{
    NegativeTemperature,
    NonPositiveMolarMass,
};

const gas_constant: f64 = 8.31446261815324;

/// Computes average molecular speed from Maxwell-Boltzmann distribution:
/// v_avg = sqrt(8RT / (pi M)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn avgSpeedOfMolecule(temperature: f64, molar_mass: f64) GasSpeedError!f64 {
    if (temperature < 0) {
        return GasSpeedError.NegativeTemperature;
    }
    if (molar_mass <= 0) {
        return GasSpeedError.NonPositiveMolarMass;
    }

    return @sqrt((8.0 * gas_constant * temperature) / (std.math.pi * molar_mass));
}

/// Computes most probable molecular speed:
/// v_mp = sqrt(2RT / M).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn mpsSpeedOfMolecule(temperature: f64, molar_mass: f64) GasSpeedError!f64 {
    if (temperature < 0) {
        return GasSpeedError.NegativeTemperature;
    }
    if (molar_mass <= 0) {
        return GasSpeedError.NonPositiveMolarMass;
    }

    return @sqrt((2.0 * gas_constant * temperature) / molar_mass);
}

test "speeds of gas molecules: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 454.3488755062257), try avgSpeedOfMolecule(273, 0.028), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 445.5257273433045), try avgSpeedOfMolecule(300, 0.032), 1e-9);

    try testing.expectApproxEqAbs(@as(f64, 402.65620702280023), try mpsSpeedOfMolecule(273, 0.028), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 394.8368955535605), try mpsSpeedOfMolecule(300, 0.032), 1e-9);
}

test "speeds of gas molecules: validation and extreme values" {
    try testing.expectError(GasSpeedError.NegativeTemperature, avgSpeedOfMolecule(-273, 0.028));
    try testing.expectError(GasSpeedError.NonPositiveMolarMass, avgSpeedOfMolecule(273, 0));
    try testing.expectError(GasSpeedError.NegativeTemperature, mpsSpeedOfMolecule(-273, 0.028));
    try testing.expectError(GasSpeedError.NonPositiveMolarMass, mpsSpeedOfMolecule(273, 0));

    const extreme = try avgSpeedOfMolecule(1e6, 1e-8);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
