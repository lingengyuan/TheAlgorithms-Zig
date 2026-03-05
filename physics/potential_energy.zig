//! Potential Energy - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/potential_energy.py

const std = @import("std");
const testing = std.testing;

pub const PotentialEnergyError = error{
    NegativeMass,
    NegativeHeight,
};

pub const standardGravity: f64 = 9.80665;

/// Computes near-earth gravitational potential energy: U = m * g * h.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn potentialEnergy(mass: f64, height: f64) PotentialEnergyError!f64 {
    if (mass < 0) return PotentialEnergyError.NegativeMass;
    if (height < 0) return PotentialEnergyError.NegativeHeight;
    return mass * standardGravity * height;
}

test "potential energy: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 980.665), try potentialEnergy(10, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try potentialEnergy(0, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try potentialEnergy(8, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 490.3325), try potentialEnergy(10, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try potentialEnergy(0, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 156.9064), try potentialEnergy(2, 8), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 19613.3), try potentialEnergy(20, 100), 1e-9);
}

test "potential energy: invalid inputs" {
    try testing.expectError(PotentialEnergyError.NegativeMass, potentialEnergy(-1, 1));
    try testing.expectError(PotentialEnergyError.NegativeHeight, potentialEnergy(1, -1));
}

test "potential energy: boundary and extreme values" {
    const huge = try potentialEnergy(1e120, 1e120);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);

    try testing.expectApproxEqAbs(@as(f64, 0.0), try potentialEnergy(1e120, 0), 1e-12);
}
