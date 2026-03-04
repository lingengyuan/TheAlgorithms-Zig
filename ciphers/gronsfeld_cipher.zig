//! Gronsfeld Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/gronsfeld_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const GronsfeldError = error{
    EmptyKey,
    InvalidKeyCharacter,
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn parseKey(allocator: Allocator, key: []const u8) ![]u8 {
    if (key.len == 0) return GronsfeldError.EmptyKey;

    const out = try allocator.alloc(u8, key.len);
    errdefer allocator.free(out);

    for (key, 0..) |ch, i| {
        if (!std.ascii.isDigit(ch)) return GronsfeldError.InvalidKeyCharacter;
        out[i] = ch - '0';
    }

    return out;
}

fn translate(allocator: Allocator, text: []const u8, key: []const u8, encrypt_mode: bool) ![]u8 {
    const key_digits = try parseKey(allocator, key);
    defer allocator.free(key_digits);

    const out = try allocator.alloc(u8, text.len);
    errdefer allocator.free(out);

    for (text, 0..) |raw_ch, i| {
        const ch = std.ascii.toUpper(raw_ch);
        const idx = std.mem.indexOfScalar(u8, LETTERS, ch);
        if (idx == null) {
            out[i] = ch;
            continue;
        }

        const shift = @as(i64, key_digits[i % key_digits.len]);
        const base_idx: i64 = @intCast(idx.?);
        const shifted = if (encrypt_mode)
            @mod(base_idx + shift, @as(i64, 26))
        else
            @mod(base_idx - shift, @as(i64, 26));

        out[i] = LETTERS[@intCast(shifted)];
    }

    return out;
}

/// Encrypts text using Gronsfeld digit key.
/// Python-reference behavior: output is uppercase and key index follows text index.
/// Time complexity: O(n), Space complexity: O(n)
pub fn gronsfeldEncrypt(allocator: Allocator, text: []const u8, key: []const u8) ![]u8 {
    return translate(allocator, text, key, true);
}

/// Decrypts text using Gronsfeld digit key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn gronsfeldDecrypt(allocator: Allocator, text: []const u8, key: []const u8) ![]u8 {
    return translate(allocator, text, key, false);
}

test "gronsfeld: python samples" {
    const alloc = testing.allocator;

    const a = try gronsfeldEncrypt(alloc, "hello", "412");
    defer alloc.free(a);
    try testing.expectEqualStrings("LFNPP", a);

    const b = try gronsfeldEncrypt(alloc, "hello", "123");
    defer alloc.free(b);
    try testing.expectEqualStrings("IGOMQ", b);

    const c = try gronsfeldEncrypt(alloc, "yes, ¥€$ - _!@#%?", "012");
    defer alloc.free(c);
    try testing.expectEqualStrings("YFU, ¥€$ - _!@#%?", c);
}

test "gronsfeld: empty text and key errors" {
    const alloc = testing.allocator;

    const empty = try gronsfeldEncrypt(alloc, "", "123");
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    try testing.expectError(GronsfeldError.EmptyKey, gronsfeldEncrypt(alloc, "abc", ""));
    try testing.expectError(GronsfeldError.InvalidKeyCharacter, gronsfeldEncrypt(alloc, "abc", "a1"));
}

test "gronsfeld: extreme long round trip" {
    const alloc = testing.allocator;
    const n: usize = 10000;

    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (plain, 0..) |*ch, i| {
        ch.* = switch (i % 5) {
            0 => 'a',
            1 => 'Z',
            2 => '-',
            3 => '7',
            else => ' ',
        };
    }

    const enc = try gronsfeldEncrypt(alloc, plain, "41290");
    defer alloc.free(enc);

    const dec = try gronsfeldDecrypt(alloc, enc, "41290");
    defer alloc.free(dec);

    var expected = try alloc.alloc(u8, plain.len);
    defer alloc.free(expected);
    for (plain, 0..) |ch, i| expected[i] = std.ascii.toUpper(ch);

    try testing.expectEqualSlices(u8, expected, dec);
}
