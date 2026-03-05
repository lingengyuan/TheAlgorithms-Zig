//! Octal to Binary - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/octal_to_binary.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidOctal };

/// Converts an octal string to binary using 3 bits per octal digit.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn octalToBinary(
    allocator: std.mem.Allocator,
    octal_number: []const u8,
) (ConversionError || std.mem.Allocator.Error)![]u8 {
    if (octal_number.len == 0) return ConversionError.EmptyInput;

    const binary = try allocator.alloc(u8, octal_number.len * 3);
    errdefer allocator.free(binary);

    for (octal_number, 0..) |digit, index| {
        if (digit < '0' or digit > '7') return ConversionError.InvalidOctal;

        const value: u8 = digit - '0';
        binary[index * 3] = if ((value & 0b100) != 0) '1' else '0';
        binary[index * 3 + 1] = if ((value & 0b010) != 0) '1' else '0';
        binary[index * 3 + 2] = if ((value & 0b001) != 0) '1' else '0';
    }

    return binary;
}

test "octal to binary: known values" {
    const alloc = testing.allocator;

    const a = try octalToBinary(alloc, "17");
    defer alloc.free(a);
    try testing.expectEqualStrings("001111", a);

    const b = try octalToBinary(alloc, "7");
    defer alloc.free(b);
    try testing.expectEqualStrings("111", b);

    const c = try octalToBinary(alloc, "01234567");
    defer alloc.free(c);
    try testing.expectEqualStrings("000001010011100101110111", c);
}

test "octal to binary: errors" {
    const alloc = testing.allocator;

    try testing.expectError(ConversionError.EmptyInput, octalToBinary(alloc, ""));
    try testing.expectError(ConversionError.InvalidOctal, octalToBinary(alloc, "Av"));
    try testing.expectError(ConversionError.InvalidOctal, octalToBinary(alloc, "@#"));
}

test "octal to binary: extreme repeated digits" {
    const alloc = testing.allocator;

    var input: [128]u8 = undefined;
    @memset(input[0..], '7');

    const out = try octalToBinary(alloc, input[0..]);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 384), out.len);
    try testing.expect(out[0] == '1');
    try testing.expect(out[out.len - 1] == '1');
}
