//! Binary to Decimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/binary_to_decimal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidBinary, Overflow };

/// Converts a binary string (e.g. "1010", "-11") to i64.
pub fn binaryToDecimal(bin: []const u8) ConversionError!i64 {
    const trimmed = std.mem.trim(u8, bin, " \t\r\n");
    if (trimmed.len == 0) return ConversionError.EmptyInput;

    var s = trimmed;
    const negative = s[0] == '-';
    if (negative) s = s[1..];
    if (s.len == 0) return ConversionError.InvalidBinary;

    const max_positive_magnitude: u64 = @intCast(std.math.maxInt(i64));
    const max_negative_magnitude: u64 = (@as(u64, 1) << 63);

    var magnitude: u64 = 0;
    for (s) |c| {
        if (c != '0' and c != '1') return ConversionError.InvalidBinary;
        const bit: u64 = c - '0';
        const mul = @mulWithOverflow(magnitude, @as(u64, 2));
        if (mul[1] != 0) return ConversionError.Overflow;
        const add = @addWithOverflow(mul[0], bit);
        if (add[1] != 0) return ConversionError.Overflow;
        magnitude = add[0];

        if (negative) {
            if (magnitude > max_negative_magnitude) return ConversionError.Overflow;
        } else if (magnitude > max_positive_magnitude) {
            return ConversionError.Overflow;
        }
    }

    if (!negative) return @intCast(magnitude);
    if (magnitude == max_negative_magnitude) return std.math.minInt(i64);
    return -@as(i64, @intCast(magnitude));
}

test "binary to decimal: known values" {
    try testing.expectEqual(@as(i64, 5), try binaryToDecimal("101"));
    try testing.expectEqual(@as(i64, 10), try binaryToDecimal("1010"));
    try testing.expectEqual(@as(i64, 0), try binaryToDecimal("0"));
    try testing.expectEqual(@as(i64, 255), try binaryToDecimal("11111111"));
}

test "binary to decimal: negative" {
    try testing.expectEqual(@as(i64, -29), try binaryToDecimal("-11101"));
}

test "binary to decimal: whitespace and signed range boundaries" {
    try testing.expectEqual(@as(i64, 10), try binaryToDecimal(" \t1010 \n"));
    try testing.expectEqual(std.math.minInt(i64), try binaryToDecimal("-1000000000000000000000000000000000000000000000000000000000000000"));
}

test "binary to decimal: errors" {
    try testing.expectError(ConversionError.EmptyInput, binaryToDecimal(""));
    try testing.expectError(ConversionError.InvalidBinary, binaryToDecimal("102"));
    try testing.expectError(ConversionError.InvalidBinary, binaryToDecimal("abc"));
    try testing.expectError(ConversionError.Overflow, binaryToDecimal("1111111111111111111111111111111111111111111111111111111111111111"));
    try testing.expectError(ConversionError.Overflow, binaryToDecimal("-1111111111111111111111111111111111111111111111111111111111111111"));
}
