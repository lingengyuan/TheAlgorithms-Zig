//! Time Conversions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/time_conversions.py

const std = @import("std");
const testing = std.testing;

pub const TimeConversionError = error{ InvalidValue, InvalidUnit };

const TimeUnit = enum {
    seconds,
    minutes,
    hours,
    days,
    weeks,
    months,
    years,
};

fn parseTimeUnit(input: []const u8) ?TimeUnit {
    if (std.ascii.eqlIgnoreCase(input, "seconds")) return .seconds;
    if (std.ascii.eqlIgnoreCase(input, "minutes")) return .minutes;
    if (std.ascii.eqlIgnoreCase(input, "hours")) return .hours;
    if (std.ascii.eqlIgnoreCase(input, "days")) return .days;
    if (std.ascii.eqlIgnoreCase(input, "weeks")) return .weeks;
    if (std.ascii.eqlIgnoreCase(input, "months")) return .months;
    if (std.ascii.eqlIgnoreCase(input, "years")) return .years;
    return null;
}

fn timeChart(unit: TimeUnit) f64 {
    return switch (unit) {
        .seconds => 1.0,
        .minutes => 60.0,
        .hours => 3600.0,
        .days => 86400.0,
        .weeks => 604800.0,
        .months => 2629800.0,
        .years => 31557600.0,
    };
}

fn round3(value: f64) f64 {
    return std.math.round(value * 1000.0) / 1000.0;
}

/// Converts time values between supported units and rounds to 3 decimals.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertTime(
    time_value: f64,
    unit_from: []const u8,
    unit_to: []const u8,
) TimeConversionError!f64 {
    if (!std.math.isFinite(time_value) or time_value < 0) return TimeConversionError.InvalidValue;

    const from = parseTimeUnit(unit_from) orelse return TimeConversionError.InvalidUnit;
    const to = parseTimeUnit(unit_to) orelse return TimeConversionError.InvalidUnit;

    const converted = time_value * timeChart(from) / timeChart(to);
    return round3(converted);
}

test "time conversions: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try convertTime(3600, "seconds", "hours"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.972), try convertTime(3500, "Seconds", "Hours"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 24.0), try convertTime(1, "DaYs", "hours"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 7200.0), try convertTime(120, "minutes", "SeCoNdS"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 14.0), try convertTime(2, "WEEKS", "days"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 30.0), try convertTime(0.5, "hours", "MINUTES"), 1e-12);
}

test "time conversions: invalid inputs" {
    try testing.expectError(TimeConversionError.InvalidValue, convertTime(-3600, "seconds", "hours"));
    try testing.expectError(TimeConversionError.InvalidValue, convertTime(std.math.nan(f64), "seconds", "hours"));
    try testing.expectError(TimeConversionError.InvalidUnit, convertTime(1, "cool", "century"));
    try testing.expectError(TimeConversionError.InvalidUnit, convertTime(1, "seconds", "hot"));
}

test "time conversions: boundary and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try convertTime(0, "seconds", "hours"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 31_557_600.0), try convertTime(1, "years", "seconds"), 1e-6);

    const huge = try convertTime(1e12, "seconds", "years");
    try testing.expect(huge > 30_000.0);
}
