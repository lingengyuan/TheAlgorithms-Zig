//! Temperature Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/temperature_conversions.py

const std = @import("std");
const testing = std.testing;

pub const Scale = enum {
    celsius,
    fahrenheit,
    kelvin,
    rankine,
};

pub const TemperatureError = error{BelowAbsoluteZero};

fn toKelvin(value: f64, from: Scale) TemperatureError!f64 {
    return switch (from) {
        .celsius => blk: {
            const k = value + 273.15;
            if (k < 0) return TemperatureError.BelowAbsoluteZero;
            break :blk k;
        },
        .fahrenheit => blk: {
            const k = (value - 32.0) * 5.0 / 9.0 + 273.15;
            if (k < 0) return TemperatureError.BelowAbsoluteZero;
            break :blk k;
        },
        .kelvin => blk: {
            if (value < 0) return TemperatureError.BelowAbsoluteZero;
            break :blk value;
        },
        .rankine => blk: {
            if (value < 0) return TemperatureError.BelowAbsoluteZero;
            break :blk value * 5.0 / 9.0;
        },
    };
}

fn fromKelvin(kelvin: f64, to: Scale) f64 {
    return switch (to) {
        .celsius => kelvin - 273.15,
        .fahrenheit => (kelvin - 273.15) * 9.0 / 5.0 + 32.0,
        .kelvin => kelvin,
        .rankine => kelvin * 9.0 / 5.0,
    };
}

/// Converts temperature between Celsius/Fahrenheit/Kelvin/Rankine.
/// Inputs below absolute zero are rejected.
/// Time complexity: O(1)
pub fn convertTemperature(value: f64, from: Scale, to: Scale) TemperatureError!f64 {
    const k = try toKelvin(value, from);
    return fromKelvin(k, to);
}

test "temperature conversion: known values" {
    try testing.expectApproxEqAbs(@as(f64, 32.0), try convertTemperature(0.0, .celsius, .fahrenheit), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 273.15), try convertTemperature(0.0, .celsius, .kelvin), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -40.0), try convertTemperature(-40.0, .fahrenheit, .celsius), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 491.67), try convertTemperature(0.0, .celsius, .rankine), 1e-9);
}

test "temperature conversion: identity" {
    try testing.expectApproxEqAbs(@as(f64, 123.456), try convertTemperature(123.456, .celsius, .celsius), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 10.0), try convertTemperature(10.0, .kelvin, .kelvin), 1e-9);
}

test "temperature conversion: absolute zero boundaries" {
    try testing.expectApproxEqAbs(@as(f64, -273.15), try convertTemperature(0.0, .kelvin, .celsius), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -459.67), try convertTemperature(0.0, .rankine, .fahrenheit), 1e-9);
}

test "temperature conversion: below absolute zero rejected" {
    try testing.expectError(TemperatureError.BelowAbsoluteZero, convertTemperature(-274.0, .celsius, .kelvin));
    try testing.expectError(TemperatureError.BelowAbsoluteZero, convertTemperature(-500.0, .fahrenheit, .celsius));
    try testing.expectError(TemperatureError.BelowAbsoluteZero, convertTemperature(-1.0, .kelvin, .celsius));
    try testing.expectError(TemperatureError.BelowAbsoluteZero, convertTemperature(-0.1, .rankine, .kelvin));
}

test "temperature conversion: extreme high values" {
    const out = try convertTemperature(1_000_000.0, .celsius, .fahrenheit);
    try testing.expectApproxEqAbs(@as(f64, 1_800_032.0), out, 1e-6);
}
