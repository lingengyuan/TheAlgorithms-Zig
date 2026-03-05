//! Mass-Energy Equivalence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/mass_energy_equivalence.py

const std = @import("std");
const testing = std.testing;

pub const MassEnergyError = error{
    NegativeMass,
    NegativeEnergy,
};

pub const speedOfLight: f64 = 299_792_458.0;

/// Computes energy from mass using Einstein's equation: E = m * c^2.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn energyFromMass(mass: f64) MassEnergyError!f64 {
    if (mass < 0) return MassEnergyError.NegativeMass;
    return mass * speedOfLight * speedOfLight;
}

/// Computes mass from energy using: m = E / c^2.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn massFromEnergy(energy: f64) MassEnergyError!f64 {
    if (energy < 0) return MassEnergyError.NegativeEnergy;
    return energy / (speedOfLight * speedOfLight);
}

test "mass energy equivalence: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.11948945063458e19), try energyFromMass(124.56), 1e6);
    try testing.expectApproxEqAbs(@as(f64, 2.8760165719578165e19), try energyFromMass(320), 1e6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try energyFromMass(0), 1e-12);
    try testing.expectError(MassEnergyError.NegativeMass, energyFromMass(-967.9));

    try testing.expectApproxEqAbs(@as(f64, 1.3859169098203872e-15), try massFromEnergy(124.56), 1e-30);
    try testing.expectApproxEqAbs(@as(f64, 3.560480179371579e-15), try massFromEnergy(320), 1e-30);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try massFromEnergy(0), 1e-12);
    try testing.expectError(MassEnergyError.NegativeEnergy, massFromEnergy(-967.9));
}

test "mass energy equivalence: boundary and extreme values" {
    const huge_energy = try energyFromMass(1e100);
    try testing.expect(std.math.isFinite(huge_energy));
    try testing.expect(huge_energy > 0);

    const huge_mass = try massFromEnergy(1e120);
    try testing.expect(std.math.isFinite(huge_mass));
    try testing.expect(huge_mass > 0);
}
