//! XOR Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/xor_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

fn effectiveKey(key: i64) u8 {
    const normalized = @mod(if (key == 0) @as(i64, 1) else key, @as(i64, 256));
    return @intCast(normalized);
}

/// Applies XOR cipher on input bytes.
/// Behavior mirrors Python reference key handling: key==0 uses 1, and key is modulo 256.
/// Time complexity: O(n), Space complexity: O(n)
pub fn applyXorCipher(allocator: Allocator, input: []const u8, key: i64) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    const k = effectiveKey(key);
    for (input, 0..) |ch, i| out[i] = ch ^ k;
    return out;
}

pub fn encryptString(allocator: Allocator, input: []const u8, key: i64) ![]u8 {
    return applyXorCipher(allocator, input, key);
}

pub fn decryptString(allocator: Allocator, input: []const u8, key: i64) ![]u8 {
    return applyXorCipher(allocator, input, key);
}

test "xor cipher: python samples" {
    const alloc = testing.allocator;

    const a = try encryptString(alloc, "hallo welt", 1);
    defer alloc.free(a);
    try testing.expectEqualStrings("i`mmn!vdmu", a);

    const b = try encryptString(alloc, "HALLO WELT", 32);
    defer alloc.free(b);
    try testing.expectEqualSlices(u8, "hallo\x00welt", b);

    const c = try encryptString(alloc, "hallo welt", 256);
    defer alloc.free(c);
    try testing.expectEqualStrings("hallo welt", c);
}

test "xor cipher: empty input" {
    const alloc = testing.allocator;
    const out = try encryptString(alloc, "", 5);
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "xor cipher: round-trip" {
    const alloc = testing.allocator;
    const input = "TheAlgorithms Zig";

    const enc = try encryptString(alloc, input, 67);
    defer alloc.free(enc);
    const dec = try decryptString(alloc, enc, 67);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, input, dec);
}

test "xor cipher: extreme key values" {
    const alloc = testing.allocator;
    const input = "edge-case";

    const enc_max = try encryptString(alloc, input, std.math.maxInt(i64));
    defer alloc.free(enc_max);
    const dec_max = try decryptString(alloc, enc_max, std.math.maxInt(i64));
    defer alloc.free(dec_max);
    try testing.expectEqualSlices(u8, input, dec_max);

    const enc_min = try encryptString(alloc, input, std.math.minInt(i64));
    defer alloc.free(enc_min);
    const dec_min = try decryptString(alloc, enc_min, std.math.minInt(i64));
    defer alloc.free(dec_min);
    try testing.expectEqualSlices(u8, input, dec_min);
}
