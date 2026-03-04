//! Autokey Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/autokey.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const AutoKeyError = error{
    PlaintextEmpty,
    CiphertextEmpty,
    KeyEmpty,
};

fn isLowerAlpha(ch: u8) bool {
    return ch >= 'a' and ch <= 'z';
}

/// Encrypts plaintext using autokey cipher.
/// Output letter cases follow Python reference (lowercase letters).
/// Time complexity: O(n), Space complexity: O(n)
pub fn encrypt(allocator: Allocator, plaintext: []const u8, key: []const u8) ![]u8 {
    if (plaintext.len == 0) return AutoKeyError.PlaintextEmpty;
    if (key.len == 0) return AutoKeyError.KeyEmpty;

    var key_stream = std.ArrayListUnmanaged(u8){};
    defer key_stream.deinit(allocator);
    try key_stream.appendSlice(allocator, key);
    try key_stream.appendSlice(allocator, plaintext);
    for (key_stream.items) |*ch| ch.* = std.ascii.toLower(ch.*);

    const plain = try allocator.alloc(u8, plaintext.len);
    defer allocator.free(plain);
    for (plaintext, 0..) |ch, i| plain[i] = std.ascii.toLower(ch);

    const out = try allocator.alloc(u8, plaintext.len);
    errdefer allocator.free(out);

    var p_idx: usize = 0;
    var k_idx: usize = 0;

    while (p_idx < plain.len) {
        const p = plain[p_idx];
        if (!isLowerAlpha(p)) {
            out[p_idx] = p;
            p_idx += 1;
        } else if (!isLowerAlpha(key_stream.items[k_idx])) {
            k_idx += 1;
        } else {
            const c = @mod((@as(i64, p) - 97) + (@as(i64, key_stream.items[k_idx]) - 97), @as(i64, 26)) + 97;
            out[p_idx] = @intCast(c);
            p_idx += 1;
            k_idx += 1;
        }
    }

    return out;
}

/// Decrypts autokey ciphertext.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decrypt(allocator: Allocator, ciphertext: []const u8, key: []const u8) ![]u8 {
    if (ciphertext.len == 0) return AutoKeyError.CiphertextEmpty;
    if (key.len == 0) return AutoKeyError.KeyEmpty;

    var key_stream = std.ArrayListUnmanaged(u8){};
    defer key_stream.deinit(allocator);
    try key_stream.appendSlice(allocator, key);
    for (key_stream.items) |*ch| ch.* = std.ascii.toLower(ch.*);

    const out = try allocator.alloc(u8, ciphertext.len);
    errdefer allocator.free(out);

    var c_idx: usize = 0;
    var k_idx: usize = 0;

    while (c_idx < ciphertext.len) {
        const c = std.ascii.toLower(ciphertext[c_idx]);
        if (!isLowerAlpha(c)) {
            out[c_idx] = c;
        } else {
            const p = @mod((@as(i64, c) - @as(i64, key_stream.items[k_idx])), @as(i64, 26)) + 97;
            out[c_idx] = @intCast(p);
            try key_stream.append(allocator, out[c_idx]);
            k_idx += 1;
        }
        c_idx += 1;
    }

    return out;
}

test "autokey: python samples" {
    const alloc = testing.allocator;

    const a = try encrypt(alloc, "hello world", "coffee");
    defer alloc.free(a);
    try testing.expectEqualStrings("jsqqs avvwo", a);

    const b = try encrypt(alloc, "coffee is good as python", "TheAlgorithms");
    defer alloc.free(b);
    try testing.expectEqualStrings("vvjfpk wj ohvp su ddylsv", b);

    const c = try decrypt(alloc, "jsqqs avvwo", "coffee");
    defer alloc.free(c);
    try testing.expectEqualStrings("hello world", c);
}

test "autokey: error cases" {
    const alloc = testing.allocator;
    try testing.expectError(AutoKeyError.PlaintextEmpty, encrypt(alloc, "", "key"));
    try testing.expectError(AutoKeyError.KeyEmpty, encrypt(alloc, "text", ""));
    try testing.expectError(AutoKeyError.CiphertextEmpty, decrypt(alloc, "", "key"));
    try testing.expectError(AutoKeyError.KeyEmpty, decrypt(alloc, "text", ""));
}

test "autokey: preserve non letters" {
    const alloc = testing.allocator;
    const text = "hello, world! 123";
    const enc = try encrypt(alloc, text, "coffee");
    defer alloc.free(enc);
    const dec = try decrypt(alloc, enc, "coffee");
    defer alloc.free(dec);

    const expected = try alloc.alloc(u8, text.len);
    defer alloc.free(expected);
    _ = std.ascii.lowerString(expected, text);
    try testing.expectEqualStrings(expected, dec);
}

test "autokey: extreme long round-trip" {
    const alloc = testing.allocator;
    const n: usize = 10000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);
    for (0..n) |i| plain[i] = if (i % 17 == 0) ' ' else @intCast('a' + (i % 26));

    const enc = try encrypt(alloc, plain, "thealgorithms");
    defer alloc.free(enc);
    const dec = try decrypt(alloc, enc, "thealgorithms");
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, plain, dec);
}
