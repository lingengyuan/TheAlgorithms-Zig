//! Rail Fence Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/rail_fence_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CipherError = error{ InvalidKey, InvalidState };

fn railForPosition(pos: usize, key: usize) usize {
    if (key <= 1) return 0;
    const lowest = key - 1;
    const period = lowest * 2;
    const n = pos % period;
    return @min(n, period - n);
}

/// Encrypts text using rail-fence transposition.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encrypt(allocator: Allocator, input: []const u8, key_i64: i64) ![]u8 {
    if (key_i64 <= 0) return CipherError.InvalidKey;
    const key: usize = @intCast(key_i64);

    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    if (key == 1 or input.len <= key) {
        @memcpy(out, input);
        return out;
    }

    const rows = try allocator.alloc(std.ArrayListUnmanaged(u8), key);
    defer {
        for (rows) |*row| row.deinit(allocator);
        allocator.free(rows);
    }
    for (0..key) |i| rows[i] = .{};

    for (input, 0..) |ch, pos| {
        const rail = railForPosition(pos, key);
        try rows[rail].append(allocator, ch);
    }

    var cursor: usize = 0;
    for (rows) |row| {
        @memcpy(out[cursor .. cursor + row.items.len], row.items);
        cursor += row.items.len;
    }

    return out;
}

/// Decrypts rail-fence ciphertext.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decrypt(allocator: Allocator, input: []const u8, key_i64: i64) ![]u8 {
    if (key_i64 <= 0) return CipherError.InvalidKey;
    const key: usize = @intCast(key_i64);

    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    if (key == 1 or input.len <= key) {
        @memcpy(out, input);
        return out;
    }

    const counts = try allocator.alloc(usize, key);
    defer allocator.free(counts);
    @memset(counts, 0);

    for (0..input.len) |pos| {
        const rail = railForPosition(pos, key);
        counts[rail] += 1;
    }

    const rows = try allocator.alloc([]const u8, key);
    defer allocator.free(rows);

    var cursor: usize = 0;
    for (0..key) |rail| {
        rows[rail] = input[cursor .. cursor + counts[rail]];
        cursor += counts[rail];
    }

    const row_read_idx = try allocator.alloc(usize, key);
    defer allocator.free(row_read_idx);
    @memset(row_read_idx, 0);

    for (0..input.len) |pos| {
        const rail = railForPosition(pos, key);
        const idx = row_read_idx[rail];
        if (idx >= rows[rail].len) return CipherError.InvalidState;
        out[pos] = rows[rail][idx];
        row_read_idx[rail] += 1;
    }

    return out;
}

test "rail fence: python samples" {
    const alloc = testing.allocator;

    const enc = try encrypt(alloc, "Hello World", 4);
    defer alloc.free(enc);
    try testing.expectEqualStrings("HWe olordll", enc);

    const dec = try decrypt(alloc, "HWe olordll", 4);
    defer alloc.free(dec);
    try testing.expectEqualStrings("Hello World", dec);
}

test "rail fence: invalid key" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.InvalidKey, encrypt(alloc, "abc", 0));
    try testing.expectError(CipherError.InvalidKey, decrypt(alloc, "abc", -2));
}

test "rail fence: key one or huge key" {
    const alloc = testing.allocator;

    const e1 = try encrypt(alloc, "abcdef", 1);
    defer alloc.free(e1);
    try testing.expectEqualStrings("abcdef", e1);

    const d1 = try decrypt(alloc, "My key is very big", 100);
    defer alloc.free(d1);
    try testing.expectEqualStrings("My key is very big", d1);
}

test "rail fence: round-trip with spaces" {
    const alloc = testing.allocator;
    const text = "This is a message";

    const enc = try encrypt(alloc, text, 3);
    defer alloc.free(enc);
    const dec = try decrypt(alloc, enc, 3);
    defer alloc.free(dec);

    try testing.expectEqualStrings(text, dec);
}

test "rail fence: extreme long input" {
    const alloc = testing.allocator;
    const n: usize = 10_000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);
    for (0..n) |i| text[i] = if (i % 2 == 0) 'x' else 'y';

    const enc = try encrypt(alloc, text, 7);
    defer alloc.free(enc);
    const dec = try decrypt(alloc, enc, 7);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, text, dec);
}
