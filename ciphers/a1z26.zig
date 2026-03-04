//! A1Z26 Letter-Number Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/a1z26.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CipherError = error{InvalidCode};

/// Encodes each byte as `byte - 96`, matching Python reference (`ord(ch) - 96`).
/// Time complexity: O(n), Space complexity: O(n)
pub fn encode(allocator: Allocator, plain: []const u8) ![]i32 {
    const out = try allocator.alloc(i32, plain.len);
    errdefer allocator.free(out);

    for (plain, 0..) |ch, i| {
        out[i] = @as(i32, ch) - 96;
    }

    return out;
}

/// Decodes each code as `chr(code + 96)` in byte domain.
/// Returns `error.InvalidCode` when decoded value is outside byte range.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decode(allocator: Allocator, encoded: []const i32) ![]u8 {
    const out = try allocator.alloc(u8, encoded.len);
    errdefer allocator.free(out);

    for (encoded, 0..) |value, i| {
        const code = value + 96;
        if (code < 0 or code > 255) return CipherError.InvalidCode;
        out[i] = @intCast(code);
    }

    return out;
}

test "a1z26: python sample" {
    const alloc = testing.allocator;
    const encoded = try encode(alloc, "myname");
    defer alloc.free(encoded);

    try testing.expectEqual(@as(i32, 13), encoded[0]);
    try testing.expectEqual(@as(i32, 25), encoded[1]);
    try testing.expectEqual(@as(i32, 14), encoded[2]);
    try testing.expectEqual(@as(i32, 1), encoded[3]);
    try testing.expectEqual(@as(i32, 13), encoded[4]);
    try testing.expectEqual(@as(i32, 5), encoded[5]);

    const decoded = try decode(alloc, encoded);
    defer alloc.free(decoded);
    try testing.expectEqualStrings("myname", decoded);
}

test "a1z26: uppercase behavior matches formula" {
    const alloc = testing.allocator;
    const encoded = try encode(alloc, "A");
    defer alloc.free(encoded);
    try testing.expectEqual(@as(i32, -31), encoded[0]);
}

test "a1z26: empty input" {
    const alloc = testing.allocator;
    const e = try encode(alloc, "");
    defer alloc.free(e);
    try testing.expectEqual(@as(usize, 0), e.len);

    const d = try decode(alloc, &[_]i32{});
    defer alloc.free(d);
    try testing.expectEqual(@as(usize, 0), d.len);
}

test "a1z26: invalid decode value" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.InvalidCode, decode(alloc, &[_]i32{-500}));
}

test "a1z26: extreme long round-trip lowercase" {
    const alloc = testing.allocator;
    const n: usize = 5000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (0..n) |i| plain[i] = @intCast('a' + (i % 26));

    const encoded = try encode(alloc, plain);
    defer alloc.free(encoded);
    const decoded = try decode(alloc, encoded);
    defer alloc.free(decoded);

    try testing.expectEqualSlices(u8, plain, decoded);
}
