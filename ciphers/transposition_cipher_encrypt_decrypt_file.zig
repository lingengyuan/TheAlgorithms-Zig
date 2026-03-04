//! Transposition Cipher File Wrapper - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/transposition_cipher_encrypt_decrypt_file.py

const std = @import("std");
const testing = std.testing;
const transposition = @import("transposition_cipher.zig");

/// Encrypts file content with transposition cipher key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encryptContent(allocator: std.mem.Allocator, key: i64, content: []const u8) ![]u8 {
    return transposition.encryptMessage(allocator, key, content);
}

/// Decrypts file content with transposition cipher key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decryptContent(allocator: std.mem.Allocator, key: i64, content: []const u8) ![]u8 {
    return transposition.decryptMessage(allocator, key, content);
}

test "transposition file wrapper: python transposition sample" {
    const alloc = testing.allocator;

    const enc = try encryptContent(alloc, 6, "Harshil Darji");
    defer alloc.free(enc);
    try testing.expectEqualStrings("Hlia rDsahrij", enc);

    const dec = try decryptContent(alloc, 6, enc);
    defer alloc.free(dec);
    try testing.expectEqualStrings("Harshil Darji", dec);
}

test "transposition file wrapper: invalid key passthrough" {
    const alloc = testing.allocator;
    try testing.expectError(transposition.CipherError.InvalidKey, encryptContent(alloc, 0, "abc"));
}

test "transposition file wrapper: extreme long round trip" {
    const alloc = testing.allocator;

    const n: usize = 10000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);

    for (text, 0..) |*ch, i| {
        ch.* = if (i % 9 == 0) ' ' else @as(u8, @intCast('a' + (i % 26)));
    }

    const enc = try encryptContent(alloc, 37, text);
    defer alloc.free(enc);

    const dec = try decryptContent(alloc, 37, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, text, dec);
}
