//! Ohm's Law - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/ohms_law.py

const std = @import("std");
const testing = std.testing;

pub const OhmsLawError = error{
    InvalidZeroArgumentCount,
    NegativeResistance,
};

pub const OhmsLawResult = union(enum) {
    voltage: f64,
    current: f64,
    resistance: f64,
};

fn countZeros(voltage: f64, current: f64, resistance: f64) u8 {
    var count: u8 = 0;
    if (voltage == 0) count += 1;
    if (current == 0) count += 1;
    if (resistance == 0) count += 1;
    return count;
}

/// Applies Ohm's law on (voltage, current, resistance) and computes the missing one,
/// requiring exactly one input to be zero (Python-reference behavior).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn ohmsLaw(voltage: f64, current: f64, resistance: f64) OhmsLawError!OhmsLawResult {
    if (countZeros(voltage, current, resistance) != 1) {
        return OhmsLawError.InvalidZeroArgumentCount;
    }
    if (resistance < 0) return OhmsLawError.NegativeResistance;

    if (voltage == 0) {
        return OhmsLawResult{ .voltage = current * resistance };
    } else if (current == 0) {
        return OhmsLawResult{ .current = voltage / resistance };
    } else {
        return OhmsLawResult{ .resistance = voltage / current };
    }
}

test "ohms law: python examples" {
    const r1 = try ohmsLaw(10, 0, 5);
    switch (r1) {
        .current => |value| try testing.expectApproxEqAbs(@as(f64, 2.0), value, 1e-12),
        else => try testing.expect(false),
    }

    try testing.expectError(OhmsLawError.InvalidZeroArgumentCount, ohmsLaw(0, 0, 10));
    try testing.expectError(OhmsLawError.NegativeResistance, ohmsLaw(0, 1, -2));

    const r2 = try ohmsLaw(-10, 1, 0);
    switch (r2) {
        .resistance => |value| try testing.expectApproxEqAbs(@as(f64, -10.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r3 = try ohmsLaw(0, -1.5, 2);
    switch (r3) {
        .voltage => |value| try testing.expectApproxEqAbs(@as(f64, -3.0), value, 1e-12),
        else => try testing.expect(false),
    }
}

test "ohms law: boundary and extreme values" {
    const huge = try ohmsLaw(0, 1e150, 1e-100);
    switch (huge) {
        .voltage => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }

    const tiny_r = try ohmsLaw(10, 0, 1e-9);
    switch (tiny_r) {
        .current => |value| try testing.expectApproxEqAbs(@as(f64, 1e10), value, 1e-2),
        else => try testing.expect(false),
    }
}
