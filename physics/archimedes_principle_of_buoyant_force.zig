//! Archimedes Principle of Buoyant Force - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/archimedes_principle_of_buoyant_force.py

const std = @import("std");
const testing = std.testing;

pub const ArchimedesError = error{
    ImpossibleFluidDensity,
    ImpossibleObjectVolume,
    ImpossibleGravity,
};

pub const earth_gravity: f64 = 9.80665;

/// Computes buoyant force using Archimedes' principle:
/// F_b = density * volume * gravity.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn archimedesPrinciple(
    fluid_density: f64,
    volume: f64,
    gravity: f64,
) ArchimedesError!f64 {
    if (fluid_density <= 0) {
        return ArchimedesError.ImpossibleFluidDensity;
    }
    if (volume <= 0) {
        return ArchimedesError.ImpossibleObjectVolume;
    }
    if (gravity < 0) {
        return ArchimedesError.ImpossibleGravity;
    }

    return fluid_density * gravity * volume;
}

test "archimedes principle: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 19600.0), try archimedesPrinciple(500, 4, 9.8), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4885.3), try archimedesPrinciple(997, 0.5, 9.8), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6844.061035), try archimedesPrinciple(997, 0.7, earth_gravity), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try archimedesPrinciple(997, 0.7, 0), 1e-12);
}

test "archimedes principle: validation and extreme values" {
    try testing.expectError(ArchimedesError.ImpossibleObjectVolume, archimedesPrinciple(997, -0.7, earth_gravity));
    try testing.expectError(ArchimedesError.ImpossibleFluidDensity, archimedesPrinciple(0, 0.7, earth_gravity));
    try testing.expectError(ArchimedesError.ImpossibleGravity, archimedesPrinciple(997, 0.7, -9.8));

    const extreme = try archimedesPrinciple(1e20, 1e10, earth_gravity);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
