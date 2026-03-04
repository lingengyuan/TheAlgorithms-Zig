//! Vernam Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/vernam_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const VernamError = error{
    EmptyKey,
    InvalidKeyCharacter,
    InvalidTextCharacter,
};

fn alphaIndex(ch: u8) ?i64 {
    const upper = std.ascii.toUpper(ch);
    if (upper < 'A' or upper > 'Z') return null;
    return @as(i64, upper - 'A');
}

fn validateKey(key: []const u8) VernamError!void {
    if (key.len == 0) return VernamError.EmptyKey;
    for (key) |ch| {
        if (alphaIndex(ch) == null) return VernamError.InvalidKeyCharacter;
    }
}

/// Encrypts plaintext with Vernam cipher in A-Z domain.
/// Time complexity: O(n), Space complexity: O(n)
pub fn vernamEncrypt(allocator: Allocator, plaintext: []const u8, key: []const u8) ![]u8 {
    try validateKey(key);

    const out = try allocator.alloc(u8, plaintext.len);
    errdefer allocator.free(out);

    for (plaintext, 0..) |ch, i| {
        const p = alphaIndex(ch) orelse return VernamError.InvalidTextCharacter;
        const k = alphaIndex(key[i % key.len]).?;
        var ct = p + k;
        while (ct > 25) ct -= 26;
        out[i] = @as(u8, @intCast('A' + ct));
    }

    return out;
}

/// Decrypts Vernam ciphertext with repeating key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn vernamDecrypt(allocator: Allocator, ciphertext: []const u8, key: []const u8) ![]u8 {
    try validateKey(key);

    const out = try allocator.alloc(u8, ciphertext.len);
    errdefer allocator.free(out);

    for (ciphertext, 0..) |ch, i| {
        const c = alphaIndex(ch) orelse return VernamError.InvalidTextCharacter;
        const k = alphaIndex(key[i % key.len]).?;
        var p = c - k;
        while (p < 0) p += 26;
        out[i] = @as(u8, @intCast('A' + p));
    }

    return out;
}

test "vernam: python sample" {
    const alloc = testing.allocator;

    const enc = try vernamEncrypt(alloc, "HELLO", "KEY");
    defer alloc.free(enc);
    try testing.expectEqualStrings("RIJVS", enc);

    const dec = try vernamDecrypt(alloc, "RIJVS", "KEY");
    defer alloc.free(dec);
    try testing.expectEqualStrings("HELLO", dec);
}

test "vernam: key and text validation" {
    const alloc = testing.allocator;

    try testing.expectError(VernamError.EmptyKey, vernamEncrypt(alloc, "HELLO", ""));
    try testing.expectError(VernamError.InvalidKeyCharacter, vernamEncrypt(alloc, "HELLO", "K3Y"));
    try testing.expectError(VernamError.InvalidTextCharacter, vernamEncrypt(alloc, "HELLO!", "KEY"));
}

test "vernam: extreme long round trip" {
    const alloc = testing.allocator;
    const n: usize = 12000;

    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (plain, 0..) |*ch, i| {
        ch.* = @as(u8, @intCast('A' + (i % 26)));
    }

    const enc = try vernamEncrypt(alloc, plain, "ALGORITHMS");
    defer alloc.free(enc);

    const dec = try vernamDecrypt(alloc, enc, "ALGORITHMS");
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, plain, dec);
}
