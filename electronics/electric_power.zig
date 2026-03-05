//! Electric Power - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/electric_power.py

const std = @import("std");
const testing = std.testing;

pub const ElectricPowerError = error{
    InvalidZeroArgumentCount,
    NegativePower,
};

pub const ElectricPowerResult = union(enum) {
    voltage: f64,
    current: f64,
    power: f64,
};

fn countZeros(voltage: f64, current: f64, power: f64) u8 {
    var count: u8 = 0;
    if (voltage == 0) count += 1;
    if (current == 0) count += 1;
    if (power == 0) count += 1;
    return count;
}

fn roundTo2(value: f64) f64 {
    return @round(value * 100.0) / 100.0;
}

/// Computes one missing electric quantity (voltage/current/power),
/// requiring exactly one argument to be zero.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn electricPower(voltage: f64, current: f64, power: f64) ElectricPowerError!ElectricPowerResult {
    if (countZeros(voltage, current, power) != 1) return ElectricPowerError.InvalidZeroArgumentCount;
    if (power < 0) return ElectricPowerError.NegativePower;

    if (voltage == 0) {
        return ElectricPowerResult{ .voltage = power / current };
    } else if (current == 0) {
        return ElectricPowerResult{ .current = power / voltage };
    } else {
        return ElectricPowerResult{ .power = roundTo2(@abs(voltage * current)) };
    }
}

test "electric power: python examples" {
    const r1 = try electricPower(0, 2, 5);
    switch (r1) {
        .voltage => |value| try testing.expectApproxEqAbs(@as(f64, 2.5), value, 1e-12),
        else => try testing.expect(false),
    }

    const r2 = try electricPower(2, 2, 0);
    switch (r2) {
        .power => |value| try testing.expectApproxEqAbs(@as(f64, 4.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r3 = try electricPower(-2, 3, 0);
    switch (r3) {
        .power => |value| try testing.expectApproxEqAbs(@as(f64, 6.0), value, 1e-12),
        else => try testing.expect(false),
    }

    try testing.expectError(ElectricPowerError.InvalidZeroArgumentCount, electricPower(2, 4, 2));
    try testing.expectError(ElectricPowerError.InvalidZeroArgumentCount, electricPower(0, 0, 2));
    try testing.expectError(ElectricPowerError.NegativePower, electricPower(0, 2, -4));

    const r4 = try electricPower(2.2, 2.2, 0);
    switch (r4) {
        .power => |value| try testing.expectApproxEqAbs(@as(f64, 4.84), value, 1e-12),
        else => try testing.expect(false),
    }

    const r5 = try electricPower(2, 0, 6);
    switch (r5) {
        .current => |value| try testing.expectApproxEqAbs(@as(f64, 3.0), value, 1e-12),
        else => try testing.expect(false),
    }
}

test "electric power: boundary and extreme values" {
    const huge = try electricPower(1e120, 1e120, 0);
    switch (huge) {
        .power => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
