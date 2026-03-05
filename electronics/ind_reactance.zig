//! Inductive Reactance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/ind_reactance.py

const std = @import("std");
const testing = std.testing;

pub const IndReactanceError = error{
    InvalidZeroArgumentCount,
    NegativeInductance,
    NegativeFrequency,
    NegativeReactance,
};

pub const IndReactanceResult = union(enum) {
    inductance: f64,
    frequency: f64,
    reactance: f64,
};

fn countZeros(inductance: f64, frequency: f64, reactance: f64) u8 {
    var count: u8 = 0;
    if (inductance == 0) count += 1;
    if (frequency == 0) count += 1;
    if (reactance == 0) count += 1;
    return count;
}

/// Solves one missing variable from:
/// X_L = 2 * pi * f * L, with exactly one argument equal to zero.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn indReactance(inductance: f64, frequency: f64, reactance: f64) IndReactanceError!IndReactanceResult {
    if (countZeros(inductance, frequency, reactance) != 1) {
        return IndReactanceError.InvalidZeroArgumentCount;
    }
    if (inductance < 0) return IndReactanceError.NegativeInductance;
    if (frequency < 0) return IndReactanceError.NegativeFrequency;
    if (reactance < 0) return IndReactanceError.NegativeReactance;

    if (inductance == 0) {
        return IndReactanceResult{ .inductance = reactance / (2.0 * std.math.pi * frequency) };
    } else if (frequency == 0) {
        return IndReactanceResult{ .frequency = reactance / (2.0 * std.math.pi * inductance) };
    } else {
        return IndReactanceResult{ .reactance = 2.0 * std.math.pi * frequency * inductance };
    }
}

test "ind reactance: python examples" {
    try testing.expectError(IndReactanceError.NegativeInductance, indReactance(-35e-6, 1e3, 0));
    try testing.expectError(IndReactanceError.NegativeFrequency, indReactance(35e-6, -1e3, 0));
    try testing.expectError(IndReactanceError.NegativeReactance, indReactance(35e-6, 0, -1));

    const i = try indReactance(0, 10e3, 50);
    switch (i) {
        .inductance => |value| try testing.expectApproxEqAbs(@as(f64, 0.0007957747154594767), value, 1e-18),
        else => try testing.expect(false),
    }

    const f = try indReactance(35e-3, 0, 50);
    switch (f) {
        .frequency => |value| try testing.expectApproxEqAbs(@as(f64, 227.36420441699332), value, 1e-12),
        else => try testing.expect(false),
    }

    const x = try indReactance(35e-6, 1e3, 0);
    switch (x) {
        .reactance => |value| try testing.expectApproxEqAbs(@as(f64, 0.2199114857512855), value, 1e-18),
        else => try testing.expect(false),
    }
}

test "ind reactance: boundary and extreme values" {
    try testing.expectError(IndReactanceError.InvalidZeroArgumentCount, indReactance(0, 0, 10));
    try testing.expectError(IndReactanceError.InvalidZeroArgumentCount, indReactance(1, 2, 3));

    const huge = try indReactance(1e6, 1e6, 0);
    switch (huge) {
        .reactance => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
