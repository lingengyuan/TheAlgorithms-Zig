//! Coulomb's Law - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/coulombs_law.py

const std = @import("std");
const testing = std.testing;

pub const CoulombsLawError = error{
    InvalidKnownValuesCount,
    NegativeDistance,
};

pub const CoulombsLawResult = union(enum) {
    force: f64,
    charge1: f64,
    charge2: f64,
    distance: f64,
};

const coulombs_constant: f64 = 8.988e9;

fn countZeros(force: f64, charge1: f64, charge2: f64, distance: f64) u8 {
    var count: u8 = 0;
    if (force == 0) count += 1;
    if (charge1 == 0) count += 1;
    if (charge2 == 0) count += 1;
    if (distance == 0) count += 1;
    return count;
}

/// Solves one missing value in Coulomb's law given the other three.
/// Exactly one of force, charge1, charge2, distance must be zero.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn coulombsLaw(
    force: f64,
    charge1: f64,
    charge2: f64,
    distance: f64,
) CoulombsLawError!CoulombsLawResult {
    if (countZeros(force, charge1, charge2, distance) != 1) {
        return CoulombsLawError.InvalidKnownValuesCount;
    }
    if (distance < 0) {
        return CoulombsLawError.NegativeDistance;
    }

    const charge_product = @abs(charge1 * charge2);

    if (force == 0) {
        return CoulombsLawResult{ .force = coulombs_constant * charge_product / (distance * distance) };
    }
    if (charge1 == 0) {
        return CoulombsLawResult{ .charge1 = @abs(force) * (distance * distance) / (coulombs_constant * charge2) };
    }
    if (charge2 == 0) {
        return CoulombsLawResult{ .charge2 = @abs(force) * (distance * distance) / (coulombs_constant * charge1) };
    }
    return CoulombsLawResult{ .distance = @sqrt(coulombs_constant * charge_product / @abs(force)) };
}

test "coulombs law: python examples" {
    const r1 = try coulombsLaw(0, 3, 5, 2000);
    switch (r1) {
        .force => |value| try testing.expectApproxEqAbs(@as(f64, 33705.0), value, 1e-9),
        else => try testing.expect(false),
    }

    const r2 = try coulombsLaw(10, 3, 5, 0);
    switch (r2) {
        .distance => |value| try testing.expectApproxEqAbs(@as(f64, 116112.01488218177), value, 1e-9),
        else => try testing.expect(false),
    }

    const r3 = try coulombsLaw(10, 0, 5, 2000);
    switch (r3) {
        .charge1 => |value| try testing.expectApproxEqAbs(@as(f64, 0.0008900756564307966), value, 1e-15),
        else => try testing.expect(false),
    }
}

test "coulombs law: validation and edge cases" {
    try testing.expectError(CoulombsLawError.InvalidKnownValuesCount, coulombsLaw(0, 0, 5, 2000));
    try testing.expectError(CoulombsLawError.NegativeDistance, coulombsLaw(0, 3, 5, -2000));

    const negative_force_distance = try coulombsLaw(-10, 3, 5, 0);
    switch (negative_force_distance) {
        .distance => |value| try testing.expectApproxEqAbs(@as(f64, 116112.01488218177), value, 1e-9),
        else => try testing.expect(false),
    }

    const extreme = try coulombsLaw(0, 1e-9, 1e-9, 1e-12);
    switch (extreme) {
        .force => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
