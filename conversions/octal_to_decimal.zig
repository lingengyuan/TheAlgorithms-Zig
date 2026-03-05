//! Octal to Decimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/octal_to_decimal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidOctal, Overflow };

/// Converts an octal string to signed decimal integer.
/// Supports optional leading '-' and surrounding ASCII whitespace.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn octalToDecimal(input: []const u8) ConversionError!i64 {
    const trimmed = std.mem.trim(u8, input, " \t\n\r\x0B\x0C");
    if (trimmed.len == 0) return ConversionError.EmptyInput;

    var is_negative = false;
    var start: usize = 0;
    if (trimmed[0] == '-') {
        is_negative = true;
        start = 1;
    }
    if (start >= trimmed.len) return ConversionError.InvalidOctal;

    var value: i128 = 0;
    for (trimmed[start..]) |ch| {
        if (ch < '0' or ch > '7') return ConversionError.InvalidOctal;
        value = value * 8 + (ch - '0');

        if (!is_negative and value > std.math.maxInt(i64)) {
            return ConversionError.Overflow;
        }

        const min_magnitude: i128 = @as(i128, std.math.maxInt(i64)) + 1;
        if (is_negative and value > min_magnitude) {
            return ConversionError.Overflow;
        }
    }

    if (is_negative) {
        const min_magnitude: i128 = @as(i128, std.math.maxInt(i64)) + 1;
        if (value == min_magnitude) return std.math.minInt(i64);
        return @as(i64, @intCast(-value));
    }

    return @intCast(value);
}

test "octal to decimal: python examples" {
    try testing.expectEqual(@as(i64, 1), try octalToDecimal("1"));
    try testing.expectEqual(@as(i64, -1), try octalToDecimal("-1"));
    try testing.expectEqual(@as(i64, 10), try octalToDecimal("12"));
    try testing.expectEqual(@as(i64, 10), try octalToDecimal(" 12   "));
    try testing.expectEqual(@as(i64, -37), try octalToDecimal("-45"));
    try testing.expectEqual(@as(i64, 0), try octalToDecimal("0"));
    try testing.expectEqual(@as(i64, -2093), try octalToDecimal("-4055"));
}

test "octal to decimal: invalid inputs" {
    try testing.expectError(ConversionError.EmptyInput, octalToDecimal(""));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("-"));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("e"));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("8"));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("-e"));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("-8"));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("2-0Fm"));
    try testing.expectError(ConversionError.InvalidOctal, octalToDecimal("19"));
}

test "octal to decimal: i64 bounds and overflow" {
    try testing.expectEqual(std.math.maxInt(i64), try octalToDecimal("777777777777777777777"));
    try testing.expectEqual(std.math.minInt(i64), try octalToDecimal("-1000000000000000000000"));
    try testing.expectError(ConversionError.Overflow, octalToDecimal("1000000000000000000000"));
    try testing.expectError(ConversionError.Overflow, octalToDecimal("7777777777777777777777"));
}
