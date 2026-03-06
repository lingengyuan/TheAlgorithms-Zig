//! Orbital Transfer Work - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/orbital_transfer_work.py

const std = @import("std");
const testing = std.testing;

pub const OrbitalTransferWorkError = error{
    OrbitalRadiiMustBeGreaterThanZero,
};

const gravitational_constant: f64 = 6.67430e-11;

/// Computes work needed to transfer an object between circular orbits:
/// W = (G * M * m / 2) * (1/r_initial - 1/r_final).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn orbitalTransferWork(
    mass_central: f64,
    mass_object: f64,
    r_initial: f64,
    r_final: f64,
) OrbitalTransferWorkError!f64 {
    if (r_initial <= 0 or r_final <= 0) {
        return OrbitalTransferWorkError.OrbitalRadiiMustBeGreaterThanZero;
    }

    return (gravitational_constant * mass_central * mass_object / 2.0) *
        ((1.0 / r_initial) - (1.0 / r_final));
}

test "orbital transfer work: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.811e9), try orbitalTransferWork(5.972e24, 1000, 6.371e6, 7e6), 1e6);
    try testing.expectApproxEqAbs(@as(f64, -1.405e9), try orbitalTransferWork(5.972e24, 500, 7e6, 6.371e6), 1e6);
    try testing.expectApproxEqAbs(@as(f64, 1.514e11), try orbitalTransferWork(1.989e30, 1000, 1.5e11, 2.28e11), 1e8);
}

test "orbital transfer work: validation and extreme values" {
    try testing.expectError(OrbitalTransferWorkError.OrbitalRadiiMustBeGreaterThanZero, orbitalTransferWork(5.972e24, 1000, 0, 7e6));
    try testing.expectError(OrbitalTransferWorkError.OrbitalRadiiMustBeGreaterThanZero, orbitalTransferWork(5.972e24, 1000, 6.371e6, -1));

    const extreme = try orbitalTransferWork(1e30, 1e20, 1e7, 1e9);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
