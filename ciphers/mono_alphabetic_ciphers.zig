//! Mono Alphabetic Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/mono_alphabetic_ciphers.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MonoAlphabeticError = error{
    InvalidKeyLength,
    InvalidKeyCharacter,
    DuplicateKeyCharacter,
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn validateKey(key: []const u8) MonoAlphabeticError!void {
    if (key.len != 26) return MonoAlphabeticError.InvalidKeyLength;

    var seen = [_]bool{false} ** 26;
    for (key) |raw_ch| {
        const ch = std.ascii.toUpper(raw_ch);
        if (ch < 'A' or ch > 'Z') return MonoAlphabeticError.InvalidKeyCharacter;
        const idx = ch - 'A';
        if (seen[idx]) return MonoAlphabeticError.DuplicateKeyCharacter;
        seen[idx] = true;
    }
}

fn translate(allocator: Allocator, key: []const u8, message: []const u8, encrypt_mode: bool) ![]u8 {
    try validateKey(key);

    var normalized_key: [26]u8 = undefined;
    for (key, 0..) |ch, i| normalized_key[i] = std.ascii.toUpper(ch);

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (message, 0..) |symbol, i| {
        const upper = std.ascii.toUpper(symbol);
        const chars_a: []const u8 = if (encrypt_mode) &normalized_key else LETTERS;
        const chars_b: []const u8 = if (encrypt_mode) LETTERS else &normalized_key;
        const pos = std.mem.indexOfScalar(u8, chars_a, upper);

        if (pos != null) {
            const mapped = chars_b[pos.?];
            out[i] = if (std.ascii.isUpper(symbol)) mapped else std.ascii.toLower(mapped);
        } else {
            out[i] = symbol;
        }
    }

    return out;
}

/// Encrypts message with monoalphabetic substitution key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encryptMessage(allocator: Allocator, key: []const u8, message: []const u8) ![]u8 {
    return translate(allocator, key, message, true);
}

/// Decrypts message with monoalphabetic substitution key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decryptMessage(allocator: Allocator, key: []const u8, message: []const u8) ![]u8 {
    return translate(allocator, key, message, false);
}

test "mono alphabetic: python sample" {
    const alloc = testing.allocator;
    const key = "QWERTYUIOPASDFGHJKLZXCVBNM";

    const enc = try encryptMessage(alloc, key, "Hello World");
    defer alloc.free(enc);
    try testing.expectEqualStrings("Pcssi Bidsm", enc);

    const dec = try decryptMessage(alloc, key, "Hello World");
    defer alloc.free(dec);
    try testing.expectEqualStrings("Itssg Vgksr", dec);
}

test "mono alphabetic: round trip and non letters" {
    const alloc = testing.allocator;
    const key = "QWERTYUIOPASDFGHJKLZXCVBNM";

    const plain = "Attack at dawn! 123";
    const enc = try encryptMessage(alloc, key, plain);
    defer alloc.free(enc);

    const dec = try decryptMessage(alloc, key, enc);
    defer alloc.free(dec);

    try testing.expectEqualStrings(plain, dec);
}

test "mono alphabetic: invalid key" {
    const alloc = testing.allocator;

    try testing.expectError(MonoAlphabeticError.InvalidKeyLength, encryptMessage(alloc, "ABC", "HELLO"));
    try testing.expectError(MonoAlphabeticError.DuplicateKeyCharacter, encryptMessage(alloc, "QQERTYUIOPASDFGHJKLZXCVBNM", "HELLO"));
    try testing.expectError(MonoAlphabeticError.InvalidKeyCharacter, encryptMessage(alloc, "QWERTYUIOPASDFGHJKLZXCVBN1", "HELLO"));
}

test "mono alphabetic: extreme long round trip" {
    const alloc = testing.allocator;
    const key = "MNBVCXZLKJHGFDSAPOIUYTREWQ";
    const n: usize = 12000;

    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (plain, 0..) |*ch, i| {
        ch.* = switch (i % 4) {
            0 => 'A',
            1 => 'b',
            2 => ' ',
            else => '!',
        };
    }

    const enc = try encryptMessage(alloc, key, plain);
    defer alloc.free(enc);

    const dec = try decryptMessage(alloc, key, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, plain, dec);
}
