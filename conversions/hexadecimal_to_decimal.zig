//! Hexadecimal to Decimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/hexadecimal_to_decimal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidHexadecimal, Overflow };

fn hexValue(ch: u8) ConversionError!u8 {
    return switch (ch) {
        '0'...'9' => ch - '0',
        'a'...'f' => ch - 'a' + 10,
        'A'...'F' => ch - 'A' + 10,
        else => ConversionError.InvalidHexadecimal,
    };
}

/// Converts hexadecimal string to signed decimal integer.
/// Supports optional leading '-' and surrounding spaces.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn hexToDecimal(input: []const u8) ConversionError!i64 {
    const trimmed = std.mem.trim(u8, input, " \t\n\r\x0B\x0C");
    if (trimmed.len == 0) return ConversionError.EmptyInput;

    var is_negative = false;
    var start: usize = 0;
    if (trimmed[0] == '-') {
        is_negative = true;
        start = 1;
    }
    if (start >= trimmed.len) return ConversionError.InvalidHexadecimal;

    var value: i128 = 0;
    for (trimmed[start..]) |ch| {
        const digit = try hexValue(ch);
        value = value * 16 + digit;

        if (!is_negative and value > std.math.maxInt(i64)) return ConversionError.Overflow;
        const min_magnitude: i128 = @as(i128, std.math.maxInt(i64)) + 1;
        if (is_negative and value > min_magnitude) return ConversionError.Overflow;
    }

    if (is_negative) {
        const min_magnitude: i128 = @as(i128, std.math.maxInt(i64)) + 1;
        if (value == min_magnitude) return std.math.minInt(i64);
        return @as(i64, @intCast(-value));
    }

    return @intCast(value);
}

test "hexadecimal to decimal: python examples" {
    try testing.expectEqual(@as(i64, 10), try hexToDecimal("a"));
    try testing.expectEqual(@as(i64, 303), try hexToDecimal("12f"));
    try testing.expectEqual(@as(i64, 303), try hexToDecimal("   12f   "));
    try testing.expectEqual(@as(i64, 65535), try hexToDecimal("FfFf"));
    try testing.expectEqual(@as(i64, -255), try hexToDecimal("-Ff"));
}

test "hexadecimal to decimal: invalid and empty" {
    try testing.expectError(ConversionError.InvalidHexadecimal, hexToDecimal("F-f"));
    try testing.expectError(ConversionError.EmptyInput, hexToDecimal(""));
    try testing.expectError(ConversionError.InvalidHexadecimal, hexToDecimal("12m"));
}

test "hexadecimal to decimal: i64 boundaries" {
    try testing.expectEqual(std.math.maxInt(i64), try hexToDecimal("7fffffffffffffff"));
    try testing.expectEqual(std.math.minInt(i64), try hexToDecimal("-8000000000000000"));
    try testing.expectError(ConversionError.Overflow, hexToDecimal("8000000000000000"));
}
