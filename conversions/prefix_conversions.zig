//! SI and Binary Prefix Conversions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/prefix_conversions.py

const std = @import("std");
const testing = std.testing;

pub const PrefixConversionError = error{InvalidPrefix};

pub const SIUnit = enum(i8) {
    yotta = 24,
    zetta = 21,
    exa = 18,
    peta = 15,
    tera = 12,
    giga = 9,
    mega = 6,
    kilo = 3,
    hecto = 2,
    deca = 1,
    deci = -1,
    centi = -2,
    milli = -3,
    micro = -6,
    nano = -9,
    pico = -12,
    femto = -15,
    atto = -18,
    zepto = -21,
    yocto = -24,
};

pub const BinaryUnit = enum(i8) {
    yotta = 8,
    zetta = 7,
    exa = 6,
    peta = 5,
    tera = 4,
    giga = 3,
    mega = 2,
    kilo = 1,
};

fn parseSIUnit(name: []const u8) PrefixConversionError!SIUnit {
    if (std.ascii.eqlIgnoreCase(name, "yotta")) return .yotta;
    if (std.ascii.eqlIgnoreCase(name, "zetta")) return .zetta;
    if (std.ascii.eqlIgnoreCase(name, "exa")) return .exa;
    if (std.ascii.eqlIgnoreCase(name, "peta")) return .peta;
    if (std.ascii.eqlIgnoreCase(name, "tera")) return .tera;
    if (std.ascii.eqlIgnoreCase(name, "giga")) return .giga;
    if (std.ascii.eqlIgnoreCase(name, "mega")) return .mega;
    if (std.ascii.eqlIgnoreCase(name, "kilo")) return .kilo;
    if (std.ascii.eqlIgnoreCase(name, "hecto")) return .hecto;
    if (std.ascii.eqlIgnoreCase(name, "deca")) return .deca;
    if (std.ascii.eqlIgnoreCase(name, "deci")) return .deci;
    if (std.ascii.eqlIgnoreCase(name, "centi")) return .centi;
    if (std.ascii.eqlIgnoreCase(name, "milli")) return .milli;
    if (std.ascii.eqlIgnoreCase(name, "micro")) return .micro;
    if (std.ascii.eqlIgnoreCase(name, "nano")) return .nano;
    if (std.ascii.eqlIgnoreCase(name, "pico")) return .pico;
    if (std.ascii.eqlIgnoreCase(name, "femto")) return .femto;
    if (std.ascii.eqlIgnoreCase(name, "atto")) return .atto;
    if (std.ascii.eqlIgnoreCase(name, "zepto")) return .zepto;
    if (std.ascii.eqlIgnoreCase(name, "yocto")) return .yocto;
    return PrefixConversionError.InvalidPrefix;
}

fn parseBinaryUnit(name: []const u8) PrefixConversionError!BinaryUnit {
    if (std.ascii.eqlIgnoreCase(name, "yotta")) return .yotta;
    if (std.ascii.eqlIgnoreCase(name, "zetta")) return .zetta;
    if (std.ascii.eqlIgnoreCase(name, "exa")) return .exa;
    if (std.ascii.eqlIgnoreCase(name, "peta")) return .peta;
    if (std.ascii.eqlIgnoreCase(name, "tera")) return .tera;
    if (std.ascii.eqlIgnoreCase(name, "giga")) return .giga;
    if (std.ascii.eqlIgnoreCase(name, "mega")) return .mega;
    if (std.ascii.eqlIgnoreCase(name, "kilo")) return .kilo;
    return PrefixConversionError.InvalidPrefix;
}

/// Converts amount between SI prefixes.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertSiPrefix(known_amount: f64, known_prefix: SIUnit, unknown_prefix: SIUnit) f64 {
    const diff: i8 = @intFromEnum(known_prefix) - @intFromEnum(unknown_prefix);
    return known_amount * std.math.pow(f64, 10.0, @floatFromInt(diff));
}

/// Name-based SI conversion (case-insensitive).
pub fn convertSiPrefixByName(
    known_amount: f64,
    known_prefix: []const u8,
    unknown_prefix: []const u8,
) PrefixConversionError!f64 {
    return convertSiPrefix(known_amount, try parseSIUnit(known_prefix), try parseSIUnit(unknown_prefix));
}

/// Converts amount between binary prefixes.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertBinaryPrefix(known_amount: f64, known_prefix: BinaryUnit, unknown_prefix: BinaryUnit) f64 {
    const diff: i8 = @intFromEnum(known_prefix) - @intFromEnum(unknown_prefix);
    const exp = @as(i16, diff) * 10;
    return known_amount * std.math.pow(f64, 2.0, @floatFromInt(exp));
}

/// Name-based binary conversion (case-insensitive).
pub fn convertBinaryPrefixByName(
    known_amount: f64,
    known_prefix: []const u8,
    unknown_prefix: []const u8,
) PrefixConversionError!f64 {
    return convertBinaryPrefix(
        known_amount,
        try parseBinaryUnit(known_prefix),
        try parseBinaryUnit(unknown_prefix),
    );
}

test "prefix conversions: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1000.0), convertSiPrefix(1, .giga, .mega), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.001), convertSiPrefix(1, .mega, .giga), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), convertSiPrefix(1, .kilo, .kilo), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1000.0), try convertSiPrefixByName(1, "giga", "mega"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1000.0), try convertSiPrefixByName(1, "gIGa", "mEGa"), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 1024.0), convertBinaryPrefix(1, .giga, .mega), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0009765625), convertBinaryPrefix(1, .mega, .giga), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), convertBinaryPrefix(1, .kilo, .kilo), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1024.0), try convertBinaryPrefixByName(1, "giga", "mega"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1024.0), try convertBinaryPrefixByName(1, "gIGa", "mEGa"), 1e-12);
}

test "prefix conversions: invalid names" {
    try testing.expectError(PrefixConversionError.InvalidPrefix, convertSiPrefixByName(1, "bad", "mega"));
    try testing.expectError(PrefixConversionError.InvalidPrefix, convertBinaryPrefixByName(1, "mega", "bad"));
}

test "prefix conversions: extreme magnitude" {
    const to_yocto = convertSiPrefix(1, .yotta, .yocto);
    try testing.expect(to_yocto > 1e40);

    const to_kilo = convertBinaryPrefix(1, .yotta, .kilo);
    try testing.expect(to_kilo > 1e20);
}
