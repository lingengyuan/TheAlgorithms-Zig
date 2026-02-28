//! Binary to Decimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/binary_to_decimal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidBinary };

/// Converts a binary string (e.g. "1010", "-11") to i64.
pub fn binaryToDecimal(bin: []const u8) ConversionError!i64 {
    if (bin.len == 0) return ConversionError.EmptyInput;

    var s = bin;
    const negative = s[0] == '-';
    if (negative) s = s[1..];
    if (s.len == 0) return ConversionError.InvalidBinary;

    var result: i64 = 0;
    for (s) |c| {
        if (c != '0' and c != '1') return ConversionError.InvalidBinary;
        result = result * 2 + (c - '0');
    }
    return if (negative) -result else result;
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

test "binary to decimal: errors" {
    try testing.expectError(ConversionError.EmptyInput, binaryToDecimal(""));
    try testing.expectError(ConversionError.InvalidBinary, binaryToDecimal("102"));
    try testing.expectError(ConversionError.InvalidBinary, binaryToDecimal("abc"));
}
