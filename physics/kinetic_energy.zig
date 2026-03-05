//! Kinetic Energy - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/kinetic_energy.py

const std = @import("std");
const testing = std.testing;

pub const KineticEnergyError = error{
    NegativeMass,
};

/// Computes kinetic energy using the classical formula: 0.5 * m * |v|^2.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn kineticEnergy(mass: f64, velocity: f64) KineticEnergyError!f64 {
    if (mass < 0) return KineticEnergyError.NegativeMass;
    const speed = @abs(velocity);
    return 0.5 * mass * speed * speed;
}

test "kinetic energy: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 500.0), try kineticEnergy(10, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try kineticEnergy(0, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try kineticEnergy(10, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4000.0), try kineticEnergy(20, -20), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try kineticEnergy(0, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.0), try kineticEnergy(2, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 500000.0), try kineticEnergy(100, 100), 1e-12);
}

test "kinetic energy: invalid input" {
    try testing.expectError(KineticEnergyError.NegativeMass, kineticEnergy(-1, 10));
}

test "kinetic energy: boundary and extreme values" {
    const huge = try kineticEnergy(1e150, 1e75);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);

    const tiny = try kineticEnergy(1e-150, 1e-75);
    try testing.expect(std.math.isFinite(tiny));
    try testing.expect(tiny > 0);
}
