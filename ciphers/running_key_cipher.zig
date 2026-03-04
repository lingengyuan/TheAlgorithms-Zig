//! Running Key Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/running_key_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const RunningKeyError = error{
    EmptyKey,
};

fn normalizeNoSpacesUpper(allocator: Allocator, input: []const u8) ![]u8 {
    var out_list = std.ArrayListUnmanaged(u8){};
    errdefer out_list.deinit(allocator);

    for (input) |ch| {
        if (ch == ' ') continue;
        try out_list.append(allocator, std.ascii.toUpper(ch));
    }

    return try out_list.toOwnedSlice(allocator);
}

/// Encrypts plaintext with running-key cipher.
/// Python-reference behavior: spaces are removed before encryption.
/// Time complexity: O(n), Space complexity: O(n)
pub fn runningKeyEncrypt(allocator: Allocator, key: []const u8, plaintext: []const u8) ![]u8 {
    const key_clean = try normalizeNoSpacesUpper(allocator, key);
    defer allocator.free(key_clean);
    if (key_clean.len == 0) return RunningKeyError.EmptyKey;

    const plain_clean = try normalizeNoSpacesUpper(allocator, plaintext);
    defer allocator.free(plain_clean);

    const out = try allocator.alloc(u8, plain_clean.len);
    errdefer allocator.free(out);

    for (plain_clean, 0..) |ch, i| {
        const p = @as(i64, ch - 'A');
        const k = @as(i64, key_clean[i % key_clean.len] - 'A');
        const c = @mod(p + k, @as(i64, 26));
        out[i] = @as(u8, @intCast(c + 'A'));
    }

    return out;
}

/// Decrypts ciphertext with running-key cipher.
/// Python-reference behavior: spaces are removed before decryption.
/// Time complexity: O(n), Space complexity: O(n)
pub fn runningKeyDecrypt(allocator: Allocator, key: []const u8, ciphertext: []const u8) ![]u8 {
    const key_clean = try normalizeNoSpacesUpper(allocator, key);
    defer allocator.free(key_clean);
    if (key_clean.len == 0) return RunningKeyError.EmptyKey;

    const cipher_clean = try normalizeNoSpacesUpper(allocator, ciphertext);
    defer allocator.free(cipher_clean);

    const out = try allocator.alloc(u8, cipher_clean.len);
    errdefer allocator.free(out);

    for (cipher_clean, 0..) |ch, i| {
        const c = @as(i64, ch - 'A');
        const k = @as(i64, key_clean[i % key_clean.len] - 'A');
        const p = @mod(c - k, @as(i64, 26));
        out[i] = @as(u8, @intCast(p + 'A'));
    }

    return out;
}

test "running key: python sample round trip" {
    const alloc = testing.allocator;
    const key = "How does the duck know that? said Victor";

    const cipher = try runningKeyEncrypt(alloc, key, "DEFEND THIS");
    defer alloc.free(cipher);

    const plain = try runningKeyDecrypt(alloc, key, cipher);
    defer alloc.free(plain);

    try testing.expectEqualStrings("DEFENDTHIS", plain);
}

test "running key: removes spaces and uppercases" {
    const alloc = testing.allocator;

    const cipher = try runningKeyEncrypt(alloc, "a b c", "a b c");
    defer alloc.free(cipher);
    try testing.expectEqualStrings("ACE", cipher);

    const plain = try runningKeyDecrypt(alloc, "a b c", cipher);
    defer alloc.free(plain);
    try testing.expectEqualStrings("ABC", plain);
}

test "running key: invalid and extreme" {
    const alloc = testing.allocator;

    try testing.expectError(RunningKeyError.EmptyKey, runningKeyEncrypt(alloc, "   ", "ABC"));

    const n: usize = 8000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (plain, 0..) |*ch, i| {
        ch.* = if (i % 7 == 0) ' ' else @as(u8, @intCast('A' + (i % 26)));
    }

    const cipher = try runningKeyEncrypt(alloc, "LONGRUNNINGKEYTEXT", plain);
    defer alloc.free(cipher);

    const dec = try runningKeyDecrypt(alloc, "LONGRUNNINGKEYTEXT", cipher);
    defer alloc.free(dec);

    const expected = try normalizeNoSpacesUpper(alloc, plain);
    defer alloc.free(expected);

    try testing.expectEqualSlices(u8, expected, dec);
}
