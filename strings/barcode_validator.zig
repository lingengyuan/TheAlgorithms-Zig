//! Barcode Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/barcode_validator.py

const std = @import("std");
const testing = std.testing;

pub const BarcodeError = error{
    AlphabeticCharacters,
    NegativeValue,
    InvalidCharacter,
    Overflow,
};

/// Computes the EAN-style check digit from all digits except the last one.
/// Time complexity: O(d), Space complexity: O(1)
pub fn getCheckDigit(barcode: u64) u8 {
    var remaining = barcode / 10;
    var use_one = false;
    var sum: u64 = 0;

    while (remaining != 0) {
        const multiplier: u64 = if (use_one) 1 else 3;
        sum += multiplier * (remaining % 10);
        remaining /= 10;
        use_one = !use_one;
    }

    return @intCast((10 - (sum % 10)) % 10);
}

pub fn isValid(barcode: u64) bool {
    return digitCount(barcode) == 13 and getCheckDigit(barcode) == @as(u8, @intCast(barcode % 10));
}

pub fn getBarcode(text: []const u8) BarcodeError!u64 {
    if (text.len == 0) return BarcodeError.InvalidCharacter;
    if (text[0] == '-') return BarcodeError.NegativeValue;

    var has_alpha = true;
    for (text) |char| {
        if (!std.ascii.isAlphabetic(char)) {
            has_alpha = false;
            break;
        }
    }
    if (has_alpha) return BarcodeError.AlphabeticCharacters;

    var value: u64 = 0;
    for (text) |char| {
        if (!std.ascii.isDigit(char)) return BarcodeError.InvalidCharacter;
        const mul = @mulWithOverflow(value, @as(u64, 10));
        if (mul[1] != 0) return BarcodeError.Overflow;
        const add = @addWithOverflow(mul[0], @as(u64, char - '0'));
        if (add[1] != 0) return BarcodeError.Overflow;
        value = add[0];
    }
    return value;
}

fn digitCount(value: u64) usize {
    if (value == 0) return 1;
    var count: usize = 0;
    var current = value;
    while (current != 0) : (current /= 10) count += 1;
    return count;
}

test "barcode validator: python samples" {
    try testing.expectEqual(@as(u8, 9), getCheckDigit(8_718_452_538_119));
    try testing.expectEqual(@as(u8, 5), getCheckDigit(87_184_523));
    try testing.expectEqual(@as(u8, 9), getCheckDigit(87_193_425_381_086));
    try testing.expect(isValid(8_718_452_538_119));
    try testing.expect(!isValid(87_184_525));
    try testing.expect(!isValid(87_193_425_381_089));
}

test "barcode validator: parsing and invalid input" {
    try testing.expectEqual(@as(u64, 8_718_452_538_119), try getBarcode("8718452538119"));
    try testing.expectError(BarcodeError.AlphabeticCharacters, getBarcode("dwefgiweuf"));
    try testing.expectError(BarcodeError.NegativeValue, getBarcode("-123"));
    try testing.expectError(BarcodeError.InvalidCharacter, getBarcode("123-456"));
}

test "barcode validator: extreme small multiples" {
    const expected = [_]u8{ 0, 7, 4, 1, 8, 5, 2, 9, 6, 3 };
    for (expected, 0..) |digit, index| {
        try testing.expectEqual(digit, getCheckDigit(@as(u64, @intCast(index * 10))));
    }
    try testing.expect(!isValid(0));
}
