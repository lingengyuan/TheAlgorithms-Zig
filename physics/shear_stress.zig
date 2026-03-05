//! Shear Stress - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/shear_stress.py

const std = @import("std");
const testing = std.testing;

pub const ShearStressError = error{
    InvalidKnownValuesCount,
    NegativeStress,
    NegativeTangentialForce,
    NegativeArea,
};

pub const ShearStressResult = union(enum) {
    stress: f64,
    tangential_force: f64,
    area: f64,
};

fn countZeros(stress: f64, tangential_force: f64, area: f64) u8 {
    var count: u8 = 0;
    if (stress == 0) count += 1;
    if (tangential_force == 0) count += 1;
    if (area == 0) count += 1;
    return count;
}

/// Solves one unknown value among stress, tangential force, and area:
/// stress = tangential_force / area.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn shearStress(
    stress: f64,
    tangential_force: f64,
    area: f64,
) ShearStressError!ShearStressResult {
    if (countZeros(stress, tangential_force, area) != 1) {
        return ShearStressError.InvalidKnownValuesCount;
    } else if (stress < 0) {
        return ShearStressError.NegativeStress;
    } else if (tangential_force < 0) {
        return ShearStressError.NegativeTangentialForce;
    } else if (area < 0) {
        return ShearStressError.NegativeArea;
    } else if (stress == 0) {
        return ShearStressResult{ .stress = tangential_force / area };
    } else if (tangential_force == 0) {
        return ShearStressResult{ .tangential_force = stress * area };
    } else {
        return ShearStressResult{ .area = tangential_force / stress };
    }
}

test "shear stress: python examples" {
    const r1 = try shearStress(25, 100, 0);
    switch (r1) {
        .area => |value| try testing.expectApproxEqAbs(@as(f64, 4.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r2 = try shearStress(0, 1600, 200);
    switch (r2) {
        .stress => |value| try testing.expectApproxEqAbs(@as(f64, 8.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r3 = try shearStress(1000, 0, 1200);
    switch (r3) {
        .tangential_force => |value| try testing.expectApproxEqAbs(@as(f64, 1200000.0), value, 1e-6),
        else => try testing.expect(false),
    }
}

test "shear stress: validation and boundary cases" {
    try testing.expectError(ShearStressError.InvalidKnownValuesCount, shearStress(0, 0, 1));
    try testing.expectError(ShearStressError.InvalidKnownValuesCount, shearStress(1, 2, 3));
    try testing.expectError(ShearStressError.NegativeStress, shearStress(-1, 2, 0));
    try testing.expectError(ShearStressError.NegativeTangentialForce, shearStress(1, -2, 0));
    try testing.expectError(ShearStressError.NegativeArea, shearStress(1, 0, -3));
}
