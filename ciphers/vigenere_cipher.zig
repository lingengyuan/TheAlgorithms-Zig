//! Vigenere Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/vigenere_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CipherError = error{
    EmptyKey,
    InvalidKey,
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn letterIndex(ch: u8) ?i64 {
    const upper = std.ascii.toUpper(ch);
    const idx = std.mem.indexOfScalar(u8, LETTERS, upper);
    if (idx == null) return null;
    return @intCast(idx.?);
}

fn validateKey(key: []const u8) CipherError!void {
    if (key.len == 0) return CipherError.EmptyKey;
    for (key) |ch| {
        if (letterIndex(ch) == null) return CipherError.InvalidKey;
    }
}

fn translate(allocator: Allocator, key: []const u8, message: []const u8, encrypt_mode: bool) ![]u8 {
    try validateKey(key);

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    var key_index: usize = 0;
    for (message, 0..) |symbol, i| {
        const msg_idx = letterIndex(symbol);
        if (msg_idx == null) {
            out[i] = symbol;
            continue;
        }

        const key_idx = letterIndex(key[key_index]).?;
        const shifted = if (encrypt_mode)
            @mod(msg_idx.? + key_idx, @as(i64, 26))
        else
            @mod(msg_idx.? - key_idx, @as(i64, 26));

        const mapped = LETTERS[@intCast(shifted)];
        out[i] = if (std.ascii.isUpper(symbol)) mapped else std.ascii.toLower(mapped);

        key_index += 1;
        if (key_index == key.len) key_index = 0;
    }

    return out;
}

/// Encrypts Vigenere message with alphabetic key.
/// Non-letter symbols are preserved and do not consume key characters.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encryptMessage(allocator: Allocator, key: []const u8, message: []const u8) ![]u8 {
    return translate(allocator, key, message, true);
}

/// Decrypts Vigenere message with alphabetic key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decryptMessage(allocator: Allocator, key: []const u8, message: []const u8) ![]u8 {
    return translate(allocator, key, message, false);
}

test "vigenere: python encrypt sample" {
    const alloc = testing.allocator;
    const out = try encryptMessage(alloc, "HDarji", "This is Harshil Darji from Dharmaj.");
    defer alloc.free(out);
    try testing.expectEqualStrings("Akij ra Odrjqqs Gaisq muod Mphumrs.", out);
}

test "vigenere: python decrypt sample" {
    const alloc = testing.allocator;
    const out = try decryptMessage(alloc, "HDarji", "Akij ra Odrjqqs Gaisq muod Mphumrs.");
    defer alloc.free(out);
    try testing.expectEqualStrings("This is Harshil Darji from Dharmaj.", out);
}

test "vigenere: non letters preserved" {
    const alloc = testing.allocator;
    const enc = try encryptMessage(alloc, "abc", "123-!? z");
    defer alloc.free(enc);
    try testing.expectEqualStrings("123-!? z", enc);
}

test "vigenere: invalid key" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.EmptyKey, encryptMessage(alloc, "", "abc"));
    try testing.expectError(CipherError.InvalidKey, encryptMessage(alloc, "ab1", "abc"));
}

test "vigenere: extreme long message round trip" {
    const alloc = testing.allocator;
    const n: usize = 5000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);
    for (0..n) |i| msg[i] = if (i % 3 == 0) 'A' else if (i % 3 == 1) 'b' else ' ';

    const enc = try encryptMessage(alloc, "Key", msg);
    defer alloc.free(enc);
    const dec = try decryptMessage(alloc, "Key", enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, msg, dec);
}
