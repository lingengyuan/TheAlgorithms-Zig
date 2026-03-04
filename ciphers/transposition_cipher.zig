//! Transposition Cipher (Route) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/transposition_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CipherError = error{ InvalidKey, InvalidState };

/// Encrypts plaintext by columnar transposition with given key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encryptMessage(allocator: Allocator, key_i64: i64, message: []const u8) ![]u8 {
    if (key_i64 <= 0) return CipherError.InvalidKey;
    const key: usize = @intCast(key_i64);

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    if (key == 1 or message.len <= key) {
        @memcpy(out, message);
        return out;
    }

    var cursor: usize = 0;
    for (0..key) |col| {
        var pointer = col;
        while (pointer < message.len) : (pointer += key) {
            out[cursor] = message[pointer];
            cursor += 1;
        }
    }

    return out;
}

/// Decrypts transposition ciphertext with given key.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decryptMessage(allocator: Allocator, key_i64: i64, message: []const u8) ![]u8 {
    if (key_i64 <= 0) return CipherError.InvalidKey;
    const key: usize = @intCast(key_i64);

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    if (key == 1 or message.len <= key) {
        @memcpy(out, message);
        return out;
    }

    const num_cols = @divFloor(message.len + key - 1, key);
    const num_rows = key;
    const num_shaded_boxes = num_cols * num_rows - message.len;

    const plain_cols = try allocator.alloc(std.ArrayListUnmanaged(u8), num_cols);
    defer {
        for (plain_cols) |*col| col.deinit(allocator);
        allocator.free(plain_cols);
    }
    for (0..num_cols) |i| plain_cols[i] = .{};

    var col: usize = 0;
    var row: usize = 0;

    for (message) |symbol| {
        try plain_cols[col].append(allocator, symbol);
        col += 1;

        if (col == num_cols or (col == num_cols - 1 and row >= num_rows - num_shaded_boxes)) {
            col = 0;
            row += 1;
        }
    }

    var cursor: usize = 0;
    for (plain_cols) |col_buf| {
        for (col_buf.items) |ch| {
            if (cursor >= out.len) return CipherError.InvalidState;
            out[cursor] = ch;
            cursor += 1;
        }
    }

    if (cursor != out.len) return CipherError.InvalidState;
    return out;
}

test "transposition cipher: python samples" {
    const alloc = testing.allocator;

    const enc = try encryptMessage(alloc, 6, "Harshil Darji");
    defer alloc.free(enc);
    try testing.expectEqualStrings("Hlia rDsahrij", enc);

    const dec = try decryptMessage(alloc, 6, "Hlia rDsahrij");
    defer alloc.free(dec);
    try testing.expectEqualStrings("Harshil Darji", dec);
}

test "transposition cipher: invalid key" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.InvalidKey, encryptMessage(alloc, 0, "abc"));
    try testing.expectError(CipherError.InvalidKey, decryptMessage(alloc, -1, "abc"));
}

test "transposition cipher: key larger than message" {
    const alloc = testing.allocator;
    const text = "short";

    const enc = try encryptMessage(alloc, 100, text);
    defer alloc.free(enc);
    try testing.expectEqualStrings(text, enc);

    const dec = try decryptMessage(alloc, 100, text);
    defer alloc.free(dec);
    try testing.expectEqualStrings(text, dec);
}

test "transposition cipher: extreme long round-trip" {
    const alloc = testing.allocator;
    const n: usize = 9000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);
    for (0..n) |i| text[i] = if (i % 5 == 0) ' ' else @intCast('a' + (i % 26));

    const enc = try encryptMessage(alloc, 37, text);
    defer alloc.free(enc);
    const dec = try decryptMessage(alloc, 37, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, text, dec);
}
