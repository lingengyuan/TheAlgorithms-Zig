//! Electrical Impedance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/electrical_impedance.py

const std = @import("std");
const testing = std.testing;

pub const ImpedanceError = error{
    InvalidZeroArgumentCount,
    NegativeRadicand,
};

pub const ImpedanceResult = union(enum) {
    resistance: f64,
    reactance: f64,
    impedance: f64,
};

fn countZeros(resistance: f64, reactance: f64, impedance: f64) u8 {
    var count: u8 = 0;
    if (resistance == 0) count += 1;
    if (reactance == 0) count += 1;
    if (impedance == 0) count += 1;
    return count;
}

/// Solves one missing impedance component using:
/// Z^2 = R^2 + X^2.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn electricalImpedance(
    resistance: f64,
    reactance: f64,
    impedance: f64,
) ImpedanceError!ImpedanceResult {
    if (countZeros(resistance, reactance, impedance) != 1) {
        return ImpedanceError.InvalidZeroArgumentCount;
    }

    if (resistance == 0) {
        const radicand = impedance * impedance - reactance * reactance;
        if (radicand < 0) return ImpedanceError.NegativeRadicand;
        return ImpedanceResult{ .resistance = @sqrt(radicand) };
    } else if (reactance == 0) {
        const radicand = impedance * impedance - resistance * resistance;
        if (radicand < 0) return ImpedanceError.NegativeRadicand;
        return ImpedanceResult{ .reactance = @sqrt(radicand) };
    } else {
        return ImpedanceResult{ .impedance = @sqrt(resistance * resistance + reactance * reactance) };
    }
}

test "electrical impedance: python examples" {
    const r1 = try electricalImpedance(3, 4, 0);
    switch (r1) {
        .impedance => |value| try testing.expectApproxEqAbs(@as(f64, 5.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r2 = try electricalImpedance(0, 4, 5);
    switch (r2) {
        .resistance => |value| try testing.expectApproxEqAbs(@as(f64, 3.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r3 = try electricalImpedance(3, 0, 5);
    switch (r3) {
        .reactance => |value| try testing.expectApproxEqAbs(@as(f64, 4.0), value, 1e-12),
        else => try testing.expect(false),
    }

    try testing.expectError(ImpedanceError.InvalidZeroArgumentCount, electricalImpedance(3, 4, 5));
}

test "electrical impedance: boundary and extreme values" {
    try testing.expectError(ImpedanceError.NegativeRadicand, electricalImpedance(0, 10, 1));
    try testing.expectError(ImpedanceError.NegativeRadicand, electricalImpedance(10, 0, 1));

    const huge = try electricalImpedance(1e150, 1e150, 0);
    switch (huge) {
        .impedance => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
