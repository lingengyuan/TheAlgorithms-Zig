//! Base16 Encoding/Decoding - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/base16.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Base16Error = error{
    InvalidLength,
    InvalidAlphabet,
};

/// Encodes bytes to uppercase hexadecimal string.
/// Time complexity: O(n), Space complexity: O(n)
pub fn base16Encode(allocator: Allocator, data: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, data.len * 2);
    errdefer allocator.free(out);

    const hex = "0123456789ABCDEF";
    for (data, 0..) |byte, i| {
        out[2 * i] = hex[byte >> 4];
        out[2 * i + 1] = hex[byte & 0x0F];
    }
    return out;
}

/// Decodes uppercase hexadecimal string to bytes.
/// Follows Python reference validation rules (even length + uppercase hex only).
/// Time complexity: O(n), Space complexity: O(n)
pub fn base16Decode(allocator: Allocator, data: []const u8) ![]u8 {
    if (data.len % 2 != 0) return Base16Error.InvalidLength;

    for (data) |ch| {
        if (!((ch >= '0' and ch <= '9') or (ch >= 'A' and ch <= 'F'))) {
            return Base16Error.InvalidAlphabet;
        }
    }

    const out = try allocator.alloc(u8, data.len / 2);
    errdefer allocator.free(out);

    var i: usize = 0;
    while (i < data.len) : (i += 2) {
        const high = try std.fmt.charToDigit(data[i], 16);
        const low = try std.fmt.charToDigit(data[i + 1], 16);
        out[i / 2] = @intCast((high << 4) | low);
    }

    return out;
}

test "base16: python samples" {
    const alloc = testing.allocator;

    const enc = try base16Encode(alloc, "Hello World!");
    defer alloc.free(enc);
    try testing.expectEqualStrings("48656C6C6F20576F726C6421", enc);

    const dec = try base16Decode(alloc, "48656C6C6F20576F726C6421");
    defer alloc.free(dec);
    try testing.expectEqualStrings("Hello World!", dec);
}

test "base16: invalid data" {
    const alloc = testing.allocator;
    try testing.expectError(Base16Error.InvalidLength, base16Decode(alloc, "486"));
    try testing.expectError(Base16Error.InvalidAlphabet, base16Decode(alloc, "48656c"));
    try testing.expectError(Base16Error.InvalidAlphabet, base16Decode(alloc, "This is not base64 encoded data."));
}

test "base16: empty" {
    const alloc = testing.allocator;
    const enc = try base16Encode(alloc, "");
    defer alloc.free(enc);
    try testing.expectEqual(@as(usize, 0), enc.len);

    const dec = try base16Decode(alloc, "");
    defer alloc.free(dec);
    try testing.expectEqual(@as(usize, 0), dec.len);
}

test "base16: extreme round-trip" {
    const alloc = testing.allocator;
    const n: usize = 9000;
    const data = try alloc.alloc(u8, n);
    defer alloc.free(data);
    for (0..n) |i| data[i] = @intCast(i % 251);

    const enc = try base16Encode(alloc, data);
    defer alloc.free(enc);
    const dec = try base16Decode(alloc, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, data, dec);
}
