//! Octal to Hexadecimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/octal_to_hexadecimal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidOctal, Overflow };
const HEX = "0123456789ABCDEF";

/// Converts octal string to uppercase hexadecimal string with `0x` prefix.
/// Accepts optional `0o` prefix.
///
/// API note: Python reference currently returns `"0x"` (without trailing zero)
/// for input `"0"`; this implementation preserves that behavior for consistency.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn octalToHex(
    allocator: std.mem.Allocator,
    input: []const u8,
) (ConversionError || std.mem.Allocator.Error)![]u8 {
    var octal = input;
    if (std.mem.startsWith(u8, octal, "0o")) {
        octal = octal[2..];
    }

    if (octal.len == 0) return ConversionError.EmptyInput;

    var decimal: u128 = 0;
    for (octal) |ch| {
        if (ch < '0' or ch > '7') return ConversionError.InvalidOctal;

        const shifted = @shlWithOverflow(decimal, 3);
        if (shifted[1] != 0) return ConversionError.Overflow;

        const added = @addWithOverflow(shifted[0], @as(u128, ch - '0'));
        if (added[1] != 0) return ConversionError.Overflow;
        decimal = added[0];
    }

    var rev_hex = std.ArrayListUnmanaged(u8){};
    defer rev_hex.deinit(allocator);

    var value = decimal;
    while (value > 0) {
        const digit = value & 0xF;
        try rev_hex.append(allocator, HEX[@intCast(digit)]);
        value >>= 4;
    }

    const output = try allocator.alloc(u8, 2 + rev_hex.items.len);
    output[0] = '0';
    output[1] = 'x';

    for (0..rev_hex.items.len) |i| {
        output[2 + i] = rev_hex.items[rev_hex.items.len - 1 - i];
    }

    return output;
}

test "octal to hexadecimal: python examples" {
    const alloc = testing.allocator;

    const a = try octalToHex(alloc, "100");
    defer alloc.free(a);
    try testing.expectEqualStrings("0x40", a);

    const b = try octalToHex(alloc, "235");
    defer alloc.free(b);
    try testing.expectEqualStrings("0x9D", b);

    const c = try octalToHex(alloc, "0o10");
    defer alloc.free(c);
    try testing.expectEqualStrings("0x8", c);
}

test "octal to hexadecimal: invalid inputs" {
    const alloc = testing.allocator;
    try testing.expectError(ConversionError.InvalidOctal, octalToHex(alloc, "Av"));
    try testing.expectError(ConversionError.EmptyInput, octalToHex(alloc, ""));
}

test "octal to hexadecimal: zero and overflow" {
    const alloc = testing.allocator;

    const zero = try octalToHex(alloc, "0");
    defer alloc.free(zero);
    try testing.expectEqualStrings("0x", zero);

    try testing.expectError(
        ConversionError.Overflow,
        octalToHex(alloc, "777777777777777777777777777777777777777777777777777777777777"),
    );
}
