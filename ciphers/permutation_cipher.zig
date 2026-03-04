//! Permutation Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/permutation_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const PermutationError = error{
    EmptyKey,
    InvalidKeyIndex,
    DuplicateKeyIndex,
    InvalidBlockSize,
};

fn validateKey(allocator: Allocator, key: []const usize) !void {
    if (key.len == 0) return PermutationError.EmptyKey;

    const seen = try allocator.alloc(bool, key.len);
    defer allocator.free(seen);
    @memset(seen, false);

    for (key) |digit| {
        if (digit >= key.len) return PermutationError.InvalidKeyIndex;
        if (seen[digit]) return PermutationError.DuplicateKeyIndex;
        seen[digit] = true;
    }
}

/// Encrypts message using block permutation with provided key.
/// Message is uppercased to mirror Python reference behavior.
/// Time complexity: O(n), Space complexity: O(n)
pub fn permutationEncrypt(allocator: Allocator, message: []const u8, key: []const usize) ![]u8 {
    try validateKey(allocator, key);
    if (message.len % key.len != 0) return PermutationError.InvalidBlockSize;

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    var block_start: usize = 0;
    while (block_start < message.len) : (block_start += key.len) {
        for (key, 0..) |digit, j| {
            out[block_start + j] = std.ascii.toUpper(message[block_start + digit]);
        }
    }

    return out;
}

/// Decrypts message using inverse block permutation.
/// Time complexity: O(n), Space complexity: O(n)
pub fn permutationDecrypt(allocator: Allocator, encrypted_message: []const u8, key: []const usize) ![]u8 {
    try validateKey(allocator, key);
    if (encrypted_message.len % key.len != 0) return PermutationError.InvalidBlockSize;

    const out = try allocator.alloc(u8, encrypted_message.len);
    errdefer allocator.free(out);

    var block_start: usize = 0;
    while (block_start < encrypted_message.len) : (block_start += key.len) {
        for (key, 0..) |digit, j| {
            out[block_start + digit] = encrypted_message[block_start + j];
        }
    }

    return out;
}

test "permutation cipher: round trip sample" {
    const alloc = testing.allocator;
    const key = [_]usize{ 2, 0, 1, 3 };

    const enc = try permutationEncrypt(alloc, "HELLOWORLD!!", &key);
    defer alloc.free(enc);

    const dec = try permutationDecrypt(alloc, enc, &key);
    defer alloc.free(dec);

    try testing.expectEqualStrings("HELLOWORLD!!", dec);
}

test "permutation cipher: with spaces" {
    const alloc = testing.allocator;
    const key = [_]usize{ 1, 0 };

    const enc = try permutationEncrypt(alloc, "A BC", &key);
    defer alloc.free(enc);
    try testing.expectEqualStrings(" ACB", enc);

    const dec = try permutationDecrypt(alloc, enc, &key);
    defer alloc.free(dec);
    try testing.expectEqualStrings("A BC", dec);
}

test "permutation cipher: invalid key and block" {
    const alloc = testing.allocator;

    try testing.expectError(PermutationError.EmptyKey, permutationEncrypt(alloc, "AB", &[_]usize{}));
    try testing.expectError(PermutationError.InvalidKeyIndex, permutationEncrypt(alloc, "ABCD", &[_]usize{ 0, 2 }));
    try testing.expectError(PermutationError.DuplicateKeyIndex, permutationEncrypt(alloc, "ABCD", &[_]usize{ 0, 0 }));
    try testing.expectError(PermutationError.InvalidBlockSize, permutationEncrypt(alloc, "ABCDE", &[_]usize{ 0, 1 }));
}

test "permutation cipher: extreme long round trip" {
    const alloc = testing.allocator;
    const key = [_]usize{ 3, 1, 0, 2, 4 };

    const n: usize = 10000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (plain, 0..) |*ch, i| {
        ch.* = @as(u8, @intCast('A' + (i % 26)));
    }

    const enc = try permutationEncrypt(alloc, plain, &key);
    defer alloc.free(enc);

    const dec = try permutationDecrypt(alloc, enc, &key);
    defer alloc.free(dec);

    var expected = try alloc.alloc(u8, n);
    defer alloc.free(expected);
    for (plain, 0..) |ch, i| expected[i] = std.ascii.toUpper(ch);

    try testing.expectEqualSlices(u8, expected, dec);
}
