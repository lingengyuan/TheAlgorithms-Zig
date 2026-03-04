//! Simple Substitution Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/simple_substitution_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SubstitutionError = error{
    InvalidKeyLength,
    InvalidKeyCharacter,
    DuplicateKeyCharacter,
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn checkValidKey(key: []const u8) SubstitutionError!void {
    if (key.len != 26) return SubstitutionError.InvalidKeyLength;

    var seen = [_]bool{false} ** 26;
    for (key) |raw_ch| {
        const ch = std.ascii.toUpper(raw_ch);
        if (ch < 'A' or ch > 'Z') return SubstitutionError.InvalidKeyCharacter;
        const idx = ch - 'A';
        if (seen[idx]) return SubstitutionError.DuplicateKeyCharacter;
        seen[idx] = true;
    }
}

fn translateMessage(allocator: Allocator, key: []const u8, message: []const u8, decrypt_mode: bool) ![]u8 {
    try checkValidKey(key);

    var upper_key: [26]u8 = undefined;
    for (key, 0..) |ch, i| upper_key[i] = std.ascii.toUpper(ch);

    const chars_a: []const u8 = if (decrypt_mode) &upper_key else LETTERS;
    const chars_b: []const u8 = if (decrypt_mode) LETTERS else &upper_key;

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (message, 0..) |symbol, i| {
        const upper = std.ascii.toUpper(symbol);
        if (std.mem.indexOfScalar(u8, chars_a, upper)) |idx| {
            const mapped = chars_b[idx];
            out[i] = if (std.ascii.isUpper(symbol)) mapped else std.ascii.toLower(mapped);
        } else {
            out[i] = symbol;
        }
    }

    return out;
}

/// Encrypts text using substitution key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encryptMessage(allocator: Allocator, key: []const u8, message: []const u8) ![]u8 {
    return translateMessage(allocator, key, message, false);
}

/// Decrypts text using substitution key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decryptMessage(allocator: Allocator, key: []const u8, message: []const u8) ![]u8 {
    return translateMessage(allocator, key, message, true);
}

/// Generates random substitution key from `A..Z`.
/// Time complexity: O(26), Space complexity: O(26)
pub fn getRandomKey(random: std.Random) [26]u8 {
    var key = [_]u8{ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };

    var i: usize = key.len - 1;
    while (i > 0) : (i -= 1) {
        const j = random.intRangeAtMost(usize, 0, i);
        const tmp = key[i];
        key[i] = key[j];
        key[j] = tmp;
    }

    return key;
}

test "simple substitution: python doctest samples" {
    const alloc = testing.allocator;
    const key = "LFWOAYUISVKMNXPBDCRJTQEGHZ";

    const enc = try encryptMessage(alloc, key, "Harshil Darji");
    defer alloc.free(enc);
    try testing.expectEqualStrings("Ilcrism Olcvs", enc);

    const dec = try decryptMessage(alloc, key, "Ilcrism Olcvs");
    defer alloc.free(dec);
    try testing.expectEqualStrings("Harshil Darji", dec);
}

test "simple substitution: invalid key" {
    const alloc = testing.allocator;

    try testing.expectError(SubstitutionError.InvalidKeyLength, encryptMessage(alloc, "ABC", "HELLO"));
    try testing.expectError(SubstitutionError.DuplicateKeyCharacter, encryptMessage(alloc, "LFWOAYUISVKMNXPBDCRJTQEGHH", "HELLO"));
    try testing.expectError(SubstitutionError.InvalidKeyCharacter, encryptMessage(alloc, "LFWOAYUISVKMNXPBDCRJTQEGH1", "HELLO"));
}

test "simple substitution: extreme long round trip" {
    const alloc = testing.allocator;
    const key = "QWERTYUIOPASDFGHJKLZXCVBNM";

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
