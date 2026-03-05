//! Speed Conversions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/speed_conversions.py

const std = @import("std");
const testing = std.testing;

pub const SpeedConversionError = error{InvalidUnit};

const SpeedUnit = enum {
    km_h,
    m_s,
    mph,
    knot,
};

fn parseSpeedUnit(unit: []const u8) ?SpeedUnit {
    if (std.mem.eql(u8, unit, "km/h")) return .km_h;
    if (std.mem.eql(u8, unit, "m/s")) return .m_s;
    if (std.mem.eql(u8, unit, "mph")) return .mph;
    if (std.mem.eql(u8, unit, "knot")) return .knot;
    return null;
}

fn speedChart(unit: SpeedUnit) f64 {
    return switch (unit) {
        .km_h => 1.0,
        .m_s => 3.6,
        .mph => 1.609344,
        .knot => 1.852,
    };
}

fn speedChartInverse(unit: SpeedUnit) f64 {
    return switch (unit) {
        .km_h => 1.0,
        .m_s => 0.277777778,
        .mph => 0.621371192,
        .knot => 0.539956803,
    };
}

fn round3(value: f64) f64 {
    return std.math.round(value * 1000.0) / 1000.0;
}

/// Converts speed from one unit to another and rounds to 3 decimals.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertSpeed(
    speed: f64,
    unit_from: []const u8,
    unit_to: []const u8,
) SpeedConversionError!f64 {
    const from = parseSpeedUnit(unit_from) orelse return SpeedConversionError.InvalidUnit;
    const to = parseSpeedUnit(unit_to) orelse return SpeedConversionError.InvalidUnit;
    return round3(speed * speedChart(from) * speedChartInverse(to));
}

test "speed conversion: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 27.778), try convertSpeed(100, "km/h", "m/s"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 62.137), try convertSpeed(100, "km/h", "mph"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 53.996), try convertSpeed(100, "km/h", "knot"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 360.0), try convertSpeed(100, "m/s", "km/h"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 223.694), try convertSpeed(100, "m/s", "mph"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 194.384), try convertSpeed(100, "m/s", "knot"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 160.934), try convertSpeed(100, "mph", "km/h"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 44.704), try convertSpeed(100, "mph", "m/s"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 86.898), try convertSpeed(100, "mph", "knot"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 185.2), try convertSpeed(100, "knot", "km/h"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 51.444), try convertSpeed(100, "knot", "m/s"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 115.078), try convertSpeed(100, "knot", "mph"), 1e-12);
}

test "speed conversion: invalid units" {
    try testing.expectError(SpeedConversionError.InvalidUnit, convertSpeed(100, "bad", "km/h"));
    try testing.expectError(SpeedConversionError.InvalidUnit, convertSpeed(100, "km/h", "bad"));
}

test "speed conversion: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try convertSpeed(0, "km/h", "mph"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -62.137), try convertSpeed(-100, "km/h", "mph"), 1e-12);

    const huge = try convertSpeed(1e9, "m/s", "km/h");
    try testing.expect(huge > 1e9);
}
