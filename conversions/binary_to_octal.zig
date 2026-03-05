//! Binary to Octal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/binary_to_octal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidBinary };

/// Converts a binary string to octal string.
/// Left-pads zeros so length is a multiple of 3.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn binaryToOctal(
    allocator: std.mem.Allocator,
    binary: []const u8,
) (ConversionError || std.mem.Allocator.Error)![]u8 {
    if (binary.len == 0) return ConversionError.EmptyInput;

    for (binary) |ch| {
        if (ch != '0' and ch != '1') return ConversionError.InvalidBinary;
    }

    const rem = binary.len % 3;
    const pad = if (rem == 0) 0 else 3 - rem;
    const padded_len = binary.len + pad;
    const out_len = padded_len / 3;

    const out = try allocator.alloc(u8, out_len);
    errdefer allocator.free(out);

    for (0..out_len) |i| {
        const group_start = i * 3;
        var value: u8 = 0;

        for (0..3) |j| {
            const bit_index = group_start + j;
            const bit: u8 = if (bit_index < pad)
                0
            else
                binary[bit_index - pad] - '0';
            value = (value << 1) | bit;
        }

        out[i] = '0' + value;
    }

    return out;
}

test "binary to octal: known values" {
    const alloc = testing.allocator;

    const a = try binaryToOctal(alloc, "1111");
    defer alloc.free(a);
    try testing.expectEqualStrings("17", a);

    const b = try binaryToOctal(alloc, "101010101010011");
    defer alloc.free(b);
    try testing.expectEqualStrings("52523", b);

    const c = try binaryToOctal(alloc, "0");
    defer alloc.free(c);
    try testing.expectEqualStrings("0", c);
}

test "binary to octal: errors" {
    const alloc = testing.allocator;

    try testing.expectError(ConversionError.EmptyInput, binaryToOctal(alloc, ""));
    try testing.expectError(ConversionError.InvalidBinary, binaryToOctal(alloc, "a-1"));
}

test "binary to octal: extreme long input" {
    const alloc = testing.allocator;

    var bits: [300]u8 = undefined;
    @memset(bits[0..], '1');

    const out = try binaryToOctal(alloc, bits[0..]);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 100), out.len);
    for (out) |digit| {
        try testing.expect(digit == '7');
    }
}
