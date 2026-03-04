//! Base64 Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/base64_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Encodes bytes to RFC4648 base64 (with padding).
/// Time complexity: O(n), Space complexity: O(n)
pub fn base64Encode(allocator: Allocator, data: []const u8) ![]u8 {
    const size = std.base64.standard.Encoder.calcSize(data.len);
    const out = try allocator.alloc(u8, size);
    _ = std.base64.standard.Encoder.encode(out, data);
    return out;
}

/// Decodes RFC4648 base64 bytes.
/// Returns `error.InvalidPadding` or `error.InvalidCharacter` for malformed input.
/// Time complexity: O(n), Space complexity: O(n)
pub fn base64Decode(allocator: Allocator, encoded: []const u8) ![]u8 {
    const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(encoded);
    const out = try allocator.alloc(u8, decoded_size);
    errdefer allocator.free(out);

    try std.base64.standard.Decoder.decode(out, encoded);
    return out;
}

test "base64 cipher: python samples" {
    const alloc = testing.allocator;

    const a = try base64Encode(alloc, "This pull request is part of Hacktoberfest20!");
    defer alloc.free(a);
    try testing.expectEqualStrings("VGhpcyBwdWxsIHJlcXVlc3QgaXMgcGFydCBvZiBIYWNrdG9iZXJmZXN0MjAh", a);

    const b = try base64Decode(alloc, "aHR0cHM6Ly90b29scy5pZXRmLm9yZy9odG1sL3JmYzQ2NDg=");
    defer alloc.free(b);
    try testing.expectEqualStrings("https://tools.ietf.org/html/rfc4648", b);

    const c = try base64Decode(alloc, "QQ==");
    defer alloc.free(c);
    try testing.expectEqualStrings("A", c);
}

test "base64 cipher: invalid inputs" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidPadding, base64Decode(alloc, "abc"));
    try testing.expectError(error.InvalidCharacter, base64Decode(alloc, "QQ$="));
}

test "base64 cipher: empty input" {
    const alloc = testing.allocator;
    const e = try base64Encode(alloc, "");
    defer alloc.free(e);
    try testing.expectEqual(@as(usize, 0), e.len);

    const d = try base64Decode(alloc, "");
    defer alloc.free(d);
    try testing.expectEqual(@as(usize, 0), d.len);
}

test "base64 cipher: extreme round-trip" {
    const alloc = testing.allocator;
    const n: usize = 8192;
    const data = try alloc.alloc(u8, n);
    defer alloc.free(data);
    for (0..n) |i| data[i] = @intCast(i % 251);

    const encoded = try base64Encode(alloc, data);
    defer alloc.free(encoded);
    const decoded = try base64Decode(alloc, encoded);
    defer alloc.free(decoded);

    try testing.expectEqualSlices(u8, data, decoded);
}
