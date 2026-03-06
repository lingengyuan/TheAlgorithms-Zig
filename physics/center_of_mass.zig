//! Center of Mass - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/center_of_mass.py

const std = @import("std");
const testing = std.testing;

pub const CenterOfMassError = error{
    NoParticlesProvided,
    NonPositiveMass,
};

pub const Particle = struct {
    x: f64,
    y: f64,
    z: f64,
    mass: f64,
};

pub const Coord3D = struct {
    x: f64,
    y: f64,
    z: f64,
};

fn roundToTwoDecimals(value: f64) f64 {
    return @round(value * 100.0) / 100.0;
}

/// Computes center of mass for a discrete set of 3D particles.
/// Coordinates are rounded to 2 decimals to match Python behavior.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn centerOfMass(particles: []const Particle) CenterOfMassError!Coord3D {
    if (particles.len == 0) {
        return CenterOfMassError.NoParticlesProvided;
    }

    var total_mass: f64 = 0.0;
    var weighted_x: f64 = 0.0;
    var weighted_y: f64 = 0.0;
    var weighted_z: f64 = 0.0;

    for (particles) |particle| {
        if (particle.mass <= 0) {
            return CenterOfMassError.NonPositiveMass;
        }
        total_mass += particle.mass;
        weighted_x += particle.x * particle.mass;
        weighted_y += particle.y * particle.mass;
        weighted_z += particle.z * particle.mass;
    }

    return Coord3D{
        .x = roundToTwoDecimals(weighted_x / total_mass),
        .y = roundToTwoDecimals(weighted_y / total_mass),
        .z = roundToTwoDecimals(weighted_z / total_mass),
    };
}

test "center of mass: python examples" {
    const particles1 = [_]Particle{
        .{ .x = 1.5, .y = 4, .z = 3.4, .mass = 4 },
        .{ .x = 5, .y = 6.8, .z = 7, .mass = 8.1 },
        .{ .x = 9.4, .y = 10.1, .z = 11.6, .mass = 12 },
    };
    const c1 = try centerOfMass(&particles1);
    try testing.expectApproxEqAbs(@as(f64, 6.61), c1.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 7.98), c1.y, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.69), c1.z, 1e-12);

    const particles2 = [_]Particle{
        .{ .x = 1, .y = 2, .z = 3, .mass = 4 },
        .{ .x = 5, .y = 6, .z = 7, .mass = 8 },
        .{ .x = 9, .y = 10, .z = 11, .mass = 12 },
    };
    const c2 = try centerOfMass(&particles2);
    try testing.expectApproxEqAbs(@as(f64, 6.33), c2.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 7.33), c2.y, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.33), c2.z, 1e-12);
}

test "center of mass: validation and extreme values" {
    const neg_mass = [_]Particle{
        .{ .x = 1, .y = 2, .z = 3, .mass = -4 },
        .{ .x = 5, .y = 6, .z = 7, .mass = 8 },
    };
    try testing.expectError(CenterOfMassError.NonPositiveMass, centerOfMass(&neg_mass));

    const zero_mass = [_]Particle{
        .{ .x = 1, .y = 2, .z = 3, .mass = 0 },
        .{ .x = 5, .y = 6, .z = 7, .mass = 8 },
    };
    try testing.expectError(CenterOfMassError.NonPositiveMass, centerOfMass(&zero_mass));

    try testing.expectError(CenterOfMassError.NoParticlesProvided, centerOfMass(&[_]Particle{}));

    const extreme = [_]Particle{
        .{ .x = 1e100, .y = 1e100, .z = 1e100, .mass = 1e100 },
        .{ .x = 3e100, .y = 5e100, .z = 7e100, .mass = 2e100 },
    };
    const c = try centerOfMass(&extreme);
    try testing.expect(std.math.isFinite(c.x));
    try testing.expect(std.math.isFinite(c.y));
    try testing.expect(std.math.isFinite(c.z));
}
