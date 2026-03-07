//! MD5 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/md5.py

const std = @import("std");
const testing = std.testing;

const Md5 = std.crypto.hash.Md5;
const HEX_DIGITS = "0123456789abcdef";

/// Returns MD5 digest bytes for the input.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn md5(data: []const u8) [16]u8 {
    var digest: [16]u8 = undefined;
    Md5.hash(data, &digest, .{});
    return digest;
}

/// Returns lowercase hexadecimal MD5 string.
/// Caller owns the returned slice.
///
/// Time complexity: O(n)
/// Space complexity: O(1) additional plus O(32) output
pub fn md5Hex(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const digest = md5(data);
    const out = try allocator.alloc(u8, 32);
    errdefer allocator.free(out);

    for (digest, 0..) |b, i| {
        out[i * 2] = HEX_DIGITS[b >> 4];
        out[i * 2 + 1] = HEX_DIGITS[b & 0x0f];
    }
    return out;
}

fn expectMd5Hex(input: []const u8, expected: []const u8) !void {
    const alloc = testing.allocator;
    const hex = try md5Hex(alloc, input);
    defer alloc.free(hex);
    try testing.expectEqualStrings(expected, hex);
}

test "md5: python examples and vectors" {
    try expectMd5Hex("", "d41d8cd98f00b204e9800998ecf8427e");
    try expectMd5Hex("The quick brown fox jumps over the lazy dog", "9e107d9d372bb6826bd81d3542a419d6");
    try expectMd5Hex("The quick brown fox jumps over the lazy dog.", "e4d909c290d0fb1ca068ffaddf22cbd0");
}

test "md5: extreme long message" {
    const alloc = testing.allocator;
    const data = try alloc.alloc(u8, 1_000_000);
    defer alloc.free(data);
    @memset(data, 'a');
    try expectMd5Hex(data, "7707d6ae4e027c70eea2a935c2296f21");
}
