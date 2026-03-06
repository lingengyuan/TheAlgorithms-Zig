//! SHA-1 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/sha1.py

const std = @import("std");
const testing = std.testing;

const Sha1 = std.crypto.hash.Sha1;
const HEX_DIGITS = "0123456789abcdef";

/// Returns SHA-1 digest bytes for the input.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn sha1(data: []const u8) [20]u8 {
    var digest: [20]u8 = undefined;
    Sha1.hash(data, &digest, .{});
    return digest;
}

/// Returns lowercase hexadecimal SHA-1 string.
/// Caller owns the returned slice.
///
/// Time complexity: O(n)
/// Space complexity: O(1) additional plus O(40) output
pub fn sha1Hex(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const digest = sha1(data);
    const out = try allocator.alloc(u8, 40);
    errdefer allocator.free(out);

    for (digest, 0..) |b, i| {
        out[i * 2] = HEX_DIGITS[b >> 4];
        out[i * 2 + 1] = HEX_DIGITS[b & 0x0f];
    }
    return out;
}

fn expectSha1Hex(input: []const u8, expected: []const u8) !void {
    const alloc = testing.allocator;
    const hex = try sha1Hex(alloc, input);
    defer alloc.free(hex);
    try testing.expectEqualStrings(expected, hex);
}

test "sha1: python example and vectors" {
    try expectSha1Hex("Allan", "872af2d8ac3d8695387e7c804bf0e02c18df9e6e");
    try expectSha1Hex("", "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    try expectSha1Hex("abc", "a9993e364706816aba3e25717850c26c9cd0d89d");
}

test "sha1: extreme long message" {
    const alloc = testing.allocator;
    const data = try alloc.alloc(u8, 1_000_000);
    defer alloc.free(data);
    @memset(data, 'a');
    try expectSha1Hex(data, "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
}
