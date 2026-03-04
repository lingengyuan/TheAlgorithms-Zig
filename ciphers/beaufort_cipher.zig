//! Beaufort Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/beaufort_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const BeaufortError = error{
    EmptyKey,
    InvalidKeyCharacter,
    InvalidMessageCharacter,
    KeyTooShort,
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn letterIndex(ch: u8) ?i64 {
    if (ch < 'A' or ch > 'Z') return null;
    return @as(i64, ch - 'A');
}

/// Generates a repeated key with total length equal to the message length.
/// Time complexity: O(n), Space complexity: O(n)
pub fn generateKey(allocator: Allocator, message: []const u8, key: []const u8) ![]u8 {
    if (key.len == 0) return BeaufortError.EmptyKey;

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (key, 0..) |raw_ch, i| {
        const ch = std.ascii.toUpper(raw_ch);
        if (letterIndex(ch) == null) return BeaufortError.InvalidKeyCharacter;
        if (i < out.len) out[i] = ch;
    }

    for (key.len..out.len) |i| {
        out[i] = out[i % key.len];
    }

    return out;
}

/// Encrypts message using Beaufort formula from Python reference.
/// Spaces are preserved and do not consume key indexes.
/// Time complexity: O(n), Space complexity: O(n)
pub fn cipherText(allocator: Allocator, message: []const u8, key_new: []const u8) ![]u8 {
    if (key_new.len < message.len) return BeaufortError.KeyTooShort;

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    var key_i: usize = 0;
    for (message, 0..) |raw_ch, i| {
        const ch = std.ascii.toUpper(raw_ch);
        if (ch == ' ') {
            out[i] = ' ';
            continue;
        }

        const msg_idx = letterIndex(ch) orelse return BeaufortError.InvalidMessageCharacter;
        const key_idx = letterIndex(key_new[key_i]) orelse return BeaufortError.InvalidKeyCharacter;

        const x = @mod(msg_idx - key_idx, @as(i64, 26));
        out[i] = LETTERS[@intCast(x)];
        key_i += 1;
    }

    return out;
}

/// Decrypts Beaufort ciphertext with Python-reference formula.
/// Time complexity: O(n), Space complexity: O(n)
pub fn originalText(allocator: Allocator, cipher_text: []const u8, key_new: []const u8) ![]u8 {
    if (key_new.len < cipher_text.len) return BeaufortError.KeyTooShort;

    const out = try allocator.alloc(u8, cipher_text.len);
    errdefer allocator.free(out);

    var key_i: usize = 0;
    for (cipher_text, 0..) |raw_ch, i| {
        const ch = std.ascii.toUpper(raw_ch);
        if (ch == ' ') {
            out[i] = ' ';
            continue;
        }

        const enc_idx = letterIndex(ch) orelse return BeaufortError.InvalidMessageCharacter;
        const key_idx = letterIndex(key_new[key_i]) orelse return BeaufortError.InvalidKeyCharacter;

        const x = @mod(enc_idx + key_idx + 26, @as(i64, 26));
        out[i] = LETTERS[@intCast(x)];
        key_i += 1;
    }

    return out;
}

test "beaufort: python sample" {
    const alloc = testing.allocator;

    const message = "THE GERMAN ATTACK";
    const key = "SECRET";

    const generated = try generateKey(alloc, message, key);
    defer alloc.free(generated);
    try testing.expectEqualStrings("SECRETSECRETSECRE", generated);

    const enc = try cipherText(alloc, message, generated);
    defer alloc.free(enc);
    try testing.expectEqualStrings("BDC PAYUWL JPAIYI", enc);

    const dec = try originalText(alloc, enc, generated);
    defer alloc.free(dec);
    try testing.expectEqualStrings(message, dec);
}

test "beaufort: key validation and invalid message" {
    const alloc = testing.allocator;
    try testing.expectError(BeaufortError.EmptyKey, generateKey(alloc, "ABC", ""));
    try testing.expectError(BeaufortError.InvalidKeyCharacter, generateKey(alloc, "ABC", "A1"));

    const generated = try generateKey(alloc, "ABC", "KEY");
    defer alloc.free(generated);
    try testing.expectError(BeaufortError.InvalidMessageCharacter, cipherText(alloc, "AB!", generated));
}

test "beaufort: extreme long round trip" {
    const alloc = testing.allocator;
    const n: usize = 6000;

    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);
    for (msg, 0..) |*ch, i| {
        ch.* = if (i % 13 == 0) ' ' else @as(u8, @intCast('A' + (i % 26)));
    }

    const generated = try generateKey(alloc, msg, "ALGORITHM");
    defer alloc.free(generated);

    const enc = try cipherText(alloc, msg, generated);
    defer alloc.free(enc);

    const dec = try originalText(alloc, enc, generated);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, msg, dec);
}
