//! Lempel-Ziv Bitstring Decompression - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/lempel_ziv_decompress.py

const std = @import("std");
const testing = std.testing;
const lempel_ziv = @import("lempel_ziv.zig");

pub const LempelZivError = lempel_ziv.LempelZivError;

/// Decompresses a compressed bitstring following the Python reference dictionary rules.
/// Caller owns the returned slice.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn decompressData(allocator: std.mem.Allocator, data_bits: []const u8) ![]u8 {
    return lempel_ziv.decompressData(allocator, data_bits);
}

/// Removes the gamma-like size prefix used by the Python reference file format.
/// Time complexity: O(n), Space complexity: O(1)
pub fn removePrefix(data_bits: []const u8) []const u8 {
    return lempel_ziv.removePrefix(data_bits);
}

test "lempel ziv decompress: python reference examples" {
    const alloc = testing.allocator;

    const d1 = try decompressData(alloc, "00110010001110");
    defer alloc.free(d1);
    try testing.expectEqualStrings("010101010101", d1);

    const d2 = try decompressData(alloc, "110010000000110010000");
    defer alloc.free(d2);
    try testing.expectEqualStrings("11110000111100000", d2);

    try testing.expectEqualStrings("10101", removePrefix("0010110101"));
    try testing.expectEqualStrings("0110101", removePrefix("10110101"));
}

test "lempel ziv decompress: edge and extreme cases" {
    const alloc = testing.allocator;

    const empty = try decompressData(alloc, "");
    defer alloc.free(empty);
    try testing.expectEqualStrings("", empty);

    try testing.expectEqualStrings("", removePrefix(""));
    try testing.expectEqualStrings("", removePrefix("0000"));

    try testing.expectError(LempelZivError.InvalidBitCharacter, decompressData(alloc, "10a1"));
}
