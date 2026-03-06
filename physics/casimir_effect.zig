//! Casimir Effect - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/casimir_effect.py

const std = @import("std");
const testing = std.testing;

pub const CasimirEffectError = error{
    InvalidKnownValuesCount,
    NegativeForce,
    NegativeArea,
    NegativeDistance,
};

pub const CasimirResult = union(enum) {
    force: f64,
    area: f64,
    distance: f64,
};

const reduced_planck_constant: f64 = 1.054571817e-34;
const speed_of_light: f64 = 3e8;

fn countZeros(force: f64, area: f64, distance: f64) u8 {
    var count: u8 = 0;
    if (force == 0) count += 1;
    if (area == 0) count += 1;
    if (distance == 0) count += 1;
    return count;
}

/// Solves one unknown among force, area, and distance for Casimir effect:
/// F = (ħ * c * pi^2 * A) / (240 * a^4).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn casimirForce(force: f64, area: f64, distance: f64) CasimirEffectError!CasimirResult {
    if (countZeros(force, area, distance) != 1) {
        return CasimirEffectError.InvalidKnownValuesCount;
    }
    if (force < 0) {
        return CasimirEffectError.NegativeForce;
    }
    if (distance < 0) {
        return CasimirEffectError.NegativeDistance;
    }
    if (area < 0) {
        return CasimirEffectError.NegativeArea;
    }

    if (force == 0) {
        const value = (reduced_planck_constant * speed_of_light * std.math.pi * std.math.pi * area) /
            (240.0 * std.math.pow(f64, distance, 4));
        return CasimirResult{ .force = value };
    }
    if (area == 0) {
        const value = (240.0 * force * std.math.pow(f64, distance, 4)) /
            (reduced_planck_constant * speed_of_light * std.math.pi * std.math.pi);
        return CasimirResult{ .area = value };
    }

    const value = std.math.pow(
        f64,
        (reduced_planck_constant * speed_of_light * std.math.pi * std.math.pi * area) /
            (240.0 * force),
        0.25,
    );
    return CasimirResult{ .distance = value };
}

test "casimir effect: python examples" {
    const r1 = try casimirForce(0, 4, 0.03);
    switch (r1) {
        .force => |value| try testing.expectApproxEqAbs(@as(f64, 6.4248189174864216e-21), value, 1e-30),
        else => try testing.expect(false),
    }

    const r2 = try casimirForce(2635e-13, 0.0023, 0);
    switch (r2) {
        .distance => |value| try testing.expectApproxEqAbs(@as(f64, 1.0323056015031114e-05), value, 1e-16),
        else => try testing.expect(false),
    }

    const r3 = try casimirForce(2737e-21, 0, 0.0023746);
    switch (r3) {
        .area => |value| try testing.expectApproxEqAbs(@as(f64, 0.06688838837354052), value, 1e-15),
        else => try testing.expect(false),
    }
}

test "casimir effect: validation and extreme values" {
    try testing.expectError(CasimirEffectError.InvalidKnownValuesCount, casimirForce(3457e-12, 0, 0));
    try testing.expectError(CasimirEffectError.NegativeDistance, casimirForce(3457e-12, 0, -0.00344));
    try testing.expectError(CasimirEffectError.NegativeForce, casimirForce(-912e-12, 0, 0.09374));

    const extreme = try casimirForce(0, 1.0, 1e-9);
    switch (extreme) {
        .force => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
