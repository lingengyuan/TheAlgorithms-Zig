//! Hex to Binary Integer - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/hex_to_bin.py

const std = @import("std");
const testing = std.testing;

pub const HexToBinError = error{
    EmptyInput,
    InvalidValue,
    InvalidBinaryInteger,
    Overflow,
};

fn parseHexUnsigned(input: []const u8) HexToBinError!u128 {
    return std.fmt.parseInt(u128, input, 16) catch |err| switch (err) {
        error.InvalidCharacter => HexToBinError.InvalidValue,
        error.Overflow => HexToBinError.Overflow,
    };
}

/// Converts hexadecimal text into an integer composed of binary digits.
///
/// API note: this mirrors Python behavior, including that hex value `0`
/// produces an invalid binary-integer error because the generated binary string is empty.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn hexToBin(hex_num: []const u8) HexToBinError!i128 {
    const trimmed = std.mem.trim(u8, hex_num, " \t\n\r\x0B\x0C");
    if (trimmed.len == 0) return HexToBinError.EmptyInput;

    var is_negative = false;
    var digits = trimmed;
    if (trimmed[0] == '-') {
        is_negative = true;
        digits = trimmed[1..];
    }
    if (digits.len == 0) return HexToBinError.InvalidValue;

    var int_num = try parseHexUnsigned(digits);

    var value: i128 = 0;
    var place: i128 = 1;
    var has_digit = false;

    while (int_num > 0) {
        has_digit = true;
        const bit: i128 = @intCast(int_num & 1);
        if (bit == 1) {
            const add = @addWithOverflow(value, place);
            if (add[1] != 0) return HexToBinError.Overflow;
            value = add[0];
        }

        const mul = @mulWithOverflow(place, 10);
        if (mul[1] != 0) return HexToBinError.Overflow;
        place = mul[0];
        int_num >>= 1;
    }

    if (!has_digit) return HexToBinError.InvalidBinaryInteger;
    return if (is_negative) -value else value;
}

test "hex to bin: python examples" {
    try testing.expectEqual(@as(i128, 10_101_100), try hexToBin("AC"));
    try testing.expectEqual(@as(i128, 100_110_100_100), try hexToBin("9A4"));
    try testing.expectEqual(@as(i128, 100_101_111), try hexToBin("   12f   "));
    try testing.expectEqual(@as(i128, 1_111_111_111_111_111), try hexToBin("FfFf"));
    try testing.expectEqual(@as(i128, -1_111_111_111_111_111), try hexToBin("-fFfF"));
}

test "hex to bin: validation" {
    try testing.expectError(HexToBinError.InvalidValue, hexToBin("F-f"));
    try testing.expectError(HexToBinError.EmptyInput, hexToBin(""));
    try testing.expectError(HexToBinError.InvalidValue, hexToBin("-"));
}

test "hex to bin: edge and extreme behavior" {
    try testing.expectError(HexToBinError.InvalidBinaryInteger, hexToBin("0"));
    try testing.expectError(HexToBinError.Overflow, hexToBin("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"));
}
