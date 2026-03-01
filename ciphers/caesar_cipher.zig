//! Caesar Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/caesar_cipher.py

const std = @import("std");
const testing = std.testing;

pub const CipherError = error{
    EmptyAlphabet,
    DuplicateAlphabetCharacter,
};

const DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn validateAlphabet(alphabet: []const u8) CipherError!void {
    if (alphabet.len == 0) return CipherError.EmptyAlphabet;

    var seen = [_]bool{false} ** 256;
    for (alphabet) |ch| {
        if (seen[ch]) return CipherError.DuplicateAlphabetCharacter;
        seen[ch] = true;
    }
}

/// Encrypts text using Caesar cipher over the provided alphabet.
/// Characters not present in the alphabet are left unchanged.
/// Time complexity: O(n * m), where n is input length and m is alphabet length.
/// Space complexity: O(n) for the returned buffer.
pub fn encrypt(
    allocator: std.mem.Allocator,
    input: []const u8,
    key: i64,
    alphabet_opt: ?[]const u8,
) ![]u8 {
    const alphabet = alphabet_opt orelse DEFAULT_ALPHABET;
    try validateAlphabet(alphabet);

    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    const len_i64: i64 = @intCast(alphabet.len);
    const shift = @mod(key, len_i64);

    for (input, 0..) |ch, i| {
        const found = std.mem.indexOfScalar(u8, alphabet, ch);
        if (found == null) {
            out[i] = ch;
            continue;
        }

        const idx_i64: i64 = @intCast(found.?);
        const shifted = @mod(idx_i64 + shift, len_i64);
        out[i] = alphabet[@intCast(shifted)];
    }

    return out;
}

/// Decrypts text using Caesar cipher over the provided alphabet.
/// Time complexity: O(n * m), where n is input length and m is alphabet length.
/// Space complexity: O(n) for the returned buffer.
pub fn decrypt(
    allocator: std.mem.Allocator,
    input: []const u8,
    key: i64,
    alphabet_opt: ?[]const u8,
) ![]u8 {
    const alphabet = alphabet_opt orelse DEFAULT_ALPHABET;
    try validateAlphabet(alphabet);

    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    const len_i64: i64 = @intCast(alphabet.len);
    const shift = @mod(key, len_i64);

    for (input, 0..) |ch, i| {
        const found = std.mem.indexOfScalar(u8, alphabet, ch);
        if (found == null) {
            out[i] = ch;
            continue;
        }

        const idx_i64: i64 = @intCast(found.?);
        const shifted = @mod(idx_i64 - shift, len_i64);
        out[i] = alphabet[@intCast(shifted)];
    }

    return out;
}

test "caesar cipher: encrypt and decrypt default alphabet" {
    const alloc = testing.allocator;
    const plain = "The quick brown fox jumps over the lazy dog";

    const encrypted = try encrypt(alloc, plain, 8, null);
    defer alloc.free(encrypted);
    try testing.expectEqualStrings("bpm yCqks jzwEv nwF rCuxA wDmz Bpm tiHG lwo", encrypted);

    const decrypted = try decrypt(alloc, encrypted, 8, null);
    defer alloc.free(decrypted);
    try testing.expectEqualStrings(plain, decrypted);
}

test "caesar cipher: large key and custom alphabet" {
    const alloc = testing.allocator;

    const a = try encrypt(alloc, "A very large key", 8000, null);
    defer alloc.free(a);
    try testing.expectEqualStrings("s nWjq dSjYW cWq", a);

    const b = try encrypt(alloc, "a lowercase alphabet", 5, "abcdefghijklmnopqrstuvwxyz");
    defer alloc.free(b);
    try testing.expectEqualStrings("f qtbjwhfxj fqumfgjy", b);
}

test "caesar cipher: empty input and non alphabet chars" {
    const alloc = testing.allocator;

    const empty = try encrypt(alloc, "", 123, null);
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const mixed = try encrypt(alloc, "1234 !@#$", 17, null);
    defer alloc.free(mixed);
    try testing.expectEqualStrings("1234 !@#$", mixed);
}

test "caesar cipher: extreme key round trip" {
    const alloc = testing.allocator;
    const input = "Please don't brute force me!";

    const encrypted = try encrypt(alloc, input, std.math.maxInt(i64), null);
    defer alloc.free(encrypted);
    const decrypted = try decrypt(alloc, encrypted, std.math.maxInt(i64), null);
    defer alloc.free(decrypted);
    try testing.expectEqualStrings(input, decrypted);

    const encrypted_min = try encrypt(alloc, input, std.math.minInt(i64), null);
    defer alloc.free(encrypted_min);
    const decrypted_min = try decrypt(alloc, encrypted_min, std.math.minInt(i64), null);
    defer alloc.free(decrypted_min);
    try testing.expectEqualStrings(input, decrypted_min);
}

test "caesar cipher: invalid alphabets" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.EmptyAlphabet, encrypt(alloc, "abc", 1, ""));
    try testing.expectError(CipherError.DuplicateAlphabetCharacter, encrypt(alloc, "abc", 1, "abca"));
}

test "caesar cipher: fuzz round trip default alphabet" {
    return testing.fuzz({}, fuzzCaesarRoundTrip, .{});
}

fn fuzzCaesarRoundTrip(context: void, input: []const u8) anyerror!void {
    _ = context;

    const max_len = @min(input.len, @as(usize, 2048));
    const text = input[0..max_len];
    const key = deriveFuzzKey(input);

    const encrypted = try encrypt(testing.allocator, text, key, null);
    defer testing.allocator.free(encrypted);
    const decrypted = try decrypt(testing.allocator, encrypted, key, null);
    defer testing.allocator.free(decrypted);

    try testing.expectEqualSlices(u8, text, decrypted);
}

fn deriveFuzzKey(input: []const u8) i64 {
    var key: i64 = 0;
    const key_bytes_len = @min(input.len, @as(usize, 16));
    for (input[0..key_bytes_len]) |b| {
        key = key *% 257 +% @as(i64, b);
    }
    return key;
}
