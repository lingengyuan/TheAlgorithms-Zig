//! Length Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/length_conversion.py

const std = @import("std");
const testing = std.testing;

pub const LengthConversionError = error{ InvalidFromType, InvalidToType };

const Unit = enum {
    mm,
    cm,
    m,
    km,
    inch,
    ft,
    yd,
    mi,
};

fn normalizeUnitName(input: []const u8, buf: *[128]u8) ?[]const u8 {
    if (input.len > buf.len) return null;
    for (input, 0..) |char, i| {
        buf[i] = std.ascii.toLower(char);
    }

    var end = input.len;
    while (end > 0 and buf[end - 1] == 's') end -= 1; // Python: rstrip("s")
    return buf[0..end];
}

fn parseUnit(input: []const u8) ?Unit {
    var buf: [128]u8 = undefined;
    const normalized = normalizeUnitName(input, &buf) orelse return null;

    if (std.mem.eql(u8, normalized, "mm") or std.mem.eql(u8, normalized, "millimeter")) return .mm;
    if (std.mem.eql(u8, normalized, "cm") or std.mem.eql(u8, normalized, "centimeter")) return .cm;
    if (std.mem.eql(u8, normalized, "m") or std.mem.eql(u8, normalized, "meter")) return .m;
    if (std.mem.eql(u8, normalized, "km") or std.mem.eql(u8, normalized, "kilometer")) return .km;
    if (std.mem.eql(u8, normalized, "in") or std.mem.eql(u8, normalized, "inch") or std.mem.eql(u8, normalized, "inche")) return .inch;
    if (std.mem.eql(u8, normalized, "ft") or std.mem.eql(u8, normalized, "feet") or std.mem.eql(u8, normalized, "foot")) return .ft;
    if (std.mem.eql(u8, normalized, "yd") or std.mem.eql(u8, normalized, "yard")) return .yd;
    if (std.mem.eql(u8, normalized, "mi") or std.mem.eql(u8, normalized, "mile")) return .mi;
    return null;
}

fn fromFactor(unit: Unit) f64 {
    return switch (unit) {
        .mm => 0.001,
        .cm => 0.01,
        .m => 1.0,
        .km => 1000.0,
        .inch => 0.0254,
        .ft => 0.3048,
        .yd => 0.9144,
        .mi => 1609.34,
    };
}

fn toFactor(unit: Unit) f64 {
    return switch (unit) {
        .mm => 1000.0,
        .cm => 100.0,
        .m => 1.0,
        .km => 0.001,
        .inch => 39.3701,
        .ft => 3.28084,
        .yd => 1.09361,
        .mi => 0.000621371,
    };
}

/// Converts length between supported metric/imperial units.
/// Supported aliases mirror the Python reference (`meter`, `METER`, `miles`, `InChEs`, etc.).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertLength(value: f64, from_type: []const u8, to_type: []const u8) LengthConversionError!f64 {
    const from_unit = parseUnit(from_type) orelse return LengthConversionError.InvalidFromType;
    const to_unit = parseUnit(to_type) orelse return LengthConversionError.InvalidToType;
    return value * fromFactor(from_unit) * toFactor(to_unit);
}

test "length conversion: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 13.12336), try convertLength(4, "METER", "FEET"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 13.12336), try convertLength(4, "M", "FT"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.001), try convertLength(1, "meter", "kilometer"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 39370.1), try convertLength(1, "kilometer", "inch"), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.8641130000000001), try convertLength(3, "kilometer", "mile"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.6096), try convertLength(2, "feet", "meter"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.333329312), try convertLength(4, "feet", "yard"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0254), try convertLength(1, "inch", "meter"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.15656468e-05), try convertLength(2, "inch", "mile"), 1e-15);
    try testing.expectApproxEqAbs(@as(f64, 20.0), try convertLength(2, "centimeter", "millimeter"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0218722), try convertLength(2, "centimeter", "yard"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.6576), try convertLength(4, "yard", "meter"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0036576), try convertLength(4, "yard", "kilometer"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.9144000000000001), try convertLength(3, "foot", "meter"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 36.00001944), try convertLength(3, "foot", "inch"), 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 6.43736), try convertLength(4, "mile", "kilometer"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 126719.753468), try convertLength(2, "miles", "InChEs"), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.3), try convertLength(3, "millimeter", "centimeter"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.1181103), try convertLength(3, "mm", "in"), 1e-9);
}

test "length conversion: invalid unit handling" {
    try testing.expectError(LengthConversionError.InvalidFromType, convertLength(4, "wrongUnit", "inch"));
    try testing.expectError(LengthConversionError.InvalidToType, convertLength(4, "meter", "wrongUnit"));
}

test "length conversion: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try convertLength(0, "m", "ft"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -3.28084), try convertLength(-1, "meter", "feet"), 1e-9);

    const huge = try convertLength(1e9, "mile", "millimeter");
    try testing.expect(huge > 1e15);
}
