//! Newton's Law of Gravitation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/newtons_law_of_gravitation.py

const std = @import("std");
const testing = std.testing;

pub const gravitationalConstant: f64 = 6.6743e-11;

pub const GravitationError = error{
    InvalidZeroArgumentCount,
    NegativeForce,
    NegativeDistance,
    NegativeMass,
};

pub const GravitationResult = union(enum) {
    force: f64,
    mass_1: f64,
    mass_2: f64,
    distance: f64,
};

fn countZeros(force: f64, mass_1: f64, mass_2: f64, distance: f64) u8 {
    var count: u8 = 0;
    if (force == 0) count += 1;
    if (mass_1 == 0) count += 1;
    if (mass_2 == 0) count += 1;
    if (distance == 0) count += 1;
    return count;
}

/// Solves one missing variable in Newton's law:
/// F = G * m1 * m2 / d^2, with exactly one input set to zero.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn gravitationalLaw(
    force: f64,
    mass_1: f64,
    mass_2: f64,
    distance: f64,
) GravitationError!GravitationResult {
    if (countZeros(force, mass_1, mass_2, distance) != 1) {
        return GravitationError.InvalidZeroArgumentCount;
    }
    if (force < 0) return GravitationError.NegativeForce;
    if (distance < 0) return GravitationError.NegativeDistance;
    if (mass_1 < 0 or mass_2 < 0) return GravitationError.NegativeMass;

    const product_of_mass = mass_1 * mass_2;
    if (force == 0) {
        return GravitationResult{ .force = gravitationalConstant * product_of_mass / (distance * distance) };
    } else if (mass_1 == 0) {
        return GravitationResult{ .mass_1 = force * (distance * distance) / (gravitationalConstant * mass_2) };
    } else if (mass_2 == 0) {
        return GravitationResult{ .mass_2 = force * (distance * distance) / (gravitationalConstant * mass_1) };
    } else {
        return GravitationResult{ .distance = @sqrt(gravitationalConstant * product_of_mass / force) };
    }
}

test "newtons law of gravitation: python examples" {
    const r1 = try gravitationalLaw(0, 5, 10, 20);
    switch (r1) {
        .force => |value| try testing.expectApproxEqAbs(@as(f64, 8.342875e-12), value, 1e-24),
        else => try testing.expect(false),
    }

    const r2 = try gravitationalLaw(7367.382, 0, 74, 3048);
    switch (r2) {
        .mass_1 => |value| try testing.expectApproxEqRel(@as(f64, 1.385816317292268e19), value, 1e-12),
        else => try testing.expect(false),
    }

    try testing.expectError(GravitationError.InvalidZeroArgumentCount, gravitationalLaw(36337.283, 0, 0, 35584));
    try testing.expectError(GravitationError.NegativeMass, gravitationalLaw(36337.283, -674, 0, 35584));
    try testing.expectError(GravitationError.NegativeForce, gravitationalLaw(-847938e12, 674, 0, 9374));
}

test "newtons law of gravitation: boundary and extreme values" {
    const d = try gravitationalLaw(10, 5, 10, 0);
    switch (d) {
        .distance => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }

    const force_out = try gravitationalLaw(0, 1e20, 1e20, 1e10);
    switch (force_out) {
        .force => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
