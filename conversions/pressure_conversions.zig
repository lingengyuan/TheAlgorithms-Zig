//! Pressure Conversions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/pressure_conversions.py

const std = @import("std");
const testing = std.testing;

pub const PressureConversionError = error{ InvalidFromType, InvalidToType };

const Unit = enum {
    atm,
    pascal,
    bar,
    kilopascal,
    megapascal,
    psi,
    inHg,
    torr,
};

const FromTo = struct {
    from_factor: f64,
    to_factor: f64,
};

fn parseUnit(input: []const u8) ?Unit {
    if (std.mem.eql(u8, input, "atm")) return .atm;
    if (std.mem.eql(u8, input, "pascal")) return .pascal;
    if (std.mem.eql(u8, input, "bar")) return .bar;
    if (std.mem.eql(u8, input, "kilopascal")) return .kilopascal;
    if (std.mem.eql(u8, input, "megapascal")) return .megapascal;
    if (std.mem.eql(u8, input, "psi")) return .psi;
    if (std.mem.eql(u8, input, "inHg")) return .inHg;
    if (std.mem.eql(u8, input, "torr")) return .torr;
    return null;
}

fn factors(unit: Unit) FromTo {
    return switch (unit) {
        .atm => .{ .from_factor = 1.0, .to_factor = 1.0 },
        .pascal => .{ .from_factor = 0.0000098, .to_factor = 101325.0 },
        .bar => .{ .from_factor = 0.986923, .to_factor = 1.01325 },
        .kilopascal => .{ .from_factor = 0.00986923, .to_factor = 101.325 },
        .megapascal => .{ .from_factor = 9.86923, .to_factor = 0.101325 },
        .psi => .{ .from_factor = 0.068046, .to_factor = 14.6959 },
        .inHg => .{ .from_factor = 0.0334211, .to_factor = 29.9213 },
        .torr => .{ .from_factor = 0.00131579, .to_factor = 760.0 },
    };
}

/// Converts pressure values between supported units.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertPressure(
    value: f64,
    from_type: []const u8,
    to_type: []const u8,
) PressureConversionError!f64 {
    const from = parseUnit(from_type) orelse return PressureConversionError.InvalidFromType;
    const to = parseUnit(to_type) orelse return PressureConversionError.InvalidToType;
    return value * factors(from).from_factor * factors(to).to_factor;
}

test "pressure conversions: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 405300), try convertPressure(4, "atm", "pascal"), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.00014401981999999998), try convertPressure(1, "pascal", "psi"), 1e-18);
    try testing.expectApproxEqAbs(@as(f64, 0.986923), try convertPressure(1, "bar", "atm"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.029999991892499998), try convertPressure(3, "kilopascal", "bar"), 1e-15);
    try testing.expectApproxEqAbs(@as(f64, 290.074434314), try convertPressure(2, "megapascal", "psi"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 206.85984), try convertPressure(4, "psi", "torr"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0334211), try convertPressure(1, "inHg", "atm"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.019336718261000002), try convertPressure(1, "torr", "psi"), 1e-15);
}

test "pressure conversions: invalid units" {
    try testing.expectError(PressureConversionError.InvalidFromType, convertPressure(4, "wrongUnit", "atm"));
    try testing.expectError(PressureConversionError.InvalidToType, convertPressure(4, "atm", "wrongUnit"));
}

test "pressure conversions: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try convertPressure(0, "atm", "pascal"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -14.6959), try convertPressure(-1, "atm", "psi"), 1e-10);

    const huge = try convertPressure(1e8, "atm", "pascal");
    try testing.expect(huge > 1e13);
}
