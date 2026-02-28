//! Binary to Hexadecimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/binary_to_hexadecimal.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ EmptyInput, InvalidBinary };

const HEX_DIGITS = "0123456789abcdef";

/// Converts a binary string to hex string (no prefix).
/// Pads left with zeros to make length a multiple of 4.
/// Caller owns the returned slice.
pub fn binaryToHex(allocator: std.mem.Allocator, bin: []const u8) ![]u8 {
    if (bin.len == 0) return ConversionError.EmptyInput;

    // Validate
    for (bin) |c| {
        if (c != '0' and c != '1') return ConversionError.InvalidBinary;
    }

    // Pad to multiple of 4
    const rem = bin.len % 4;
    const padding = if (rem == 0) 0 else 4 - rem;
    const padded_len = bin.len + padding;
    const hex_len = padded_len / 4;

    const result = try allocator.alloc(u8, hex_len);
    errdefer allocator.free(result);

    for (0..hex_len) |h| {
        var nibble: u8 = 0;
        for (0..4) |b| {
            const bit_idx = h * 4 + b;
            const bin_idx = if (bit_idx < padding) 0 else bit_idx - padding;
            const bit: u8 = if (bit_idx < padding) 0 else (bin[bin_idx] - '0');
            nibble = (nibble << 1) | bit;
        }
        result[h] = HEX_DIGITS[@intCast(nibble)];
    }
    return result;
}

test "binary to hex: known values" {
    const alloc = testing.allocator;
    const cases = [_]struct { bin: []const u8, hex: []const u8 }{
        .{ .bin = "0", .hex = "0" },
        .{ .bin = "1111", .hex = "f" },
        .{ .bin = "11111111", .hex = "ff" },
        .{ .bin = "101011111", .hex = "15f" },
        .{ .bin = "1010", .hex = "a" },
    };
    for (cases) |c| {
        const s = try binaryToHex(alloc, c.bin);
        defer alloc.free(s);
        try testing.expectEqualStrings(c.hex, s);
    }
}

test "binary to hex: errors" {
    const alloc = testing.allocator;
    try testing.expectError(ConversionError.EmptyInput, binaryToHex(alloc, ""));
    try testing.expectError(ConversionError.InvalidBinary, binaryToHex(alloc, "102"));
}
