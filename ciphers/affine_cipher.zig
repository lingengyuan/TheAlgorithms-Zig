//! Affine Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/affine_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CipherError = error{
    WeakKeyA,
    WeakKeyB,
    InvalidKeyRange,
    NotRelativelyPrime,
    NoModInverse,
};

const SYMBOLS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

fn gcd(a_init: i64, b_init: i64) i64 {
    var a: i64 = if (a_init < 0) -a_init else a_init;
    var b: i64 = if (b_init < 0) -b_init else b_init;
    while (b != 0) {
        const t = @mod(a, b);
        a = b;
        b = t;
    }
    return a;
}

fn modInverse(a: i64, m: i64) CipherError!i64 {
    var t: i64 = 0;
    var new_t: i64 = 1;
    var r: i64 = m;
    var new_r: i64 = @mod(a, m);

    while (new_r != 0) {
        const q = @divTrunc(r, new_r);

        const next_t = t - q * new_t;
        t = new_t;
        new_t = next_t;

        const next_r = r - q * new_r;
        r = new_r;
        new_r = next_r;
    }

    if (r != 1) return CipherError.NoModInverse;
    if (t < 0) t += m;
    return t;
}

fn checkKeys(key_a: i64, key_b: i64, encrypt_mode: bool) CipherError!void {
    const m: i64 = @intCast(SYMBOLS.len);

    if (encrypt_mode) {
        if (key_a == 1) return CipherError.WeakKeyA;
        if (key_b == 0) return CipherError.WeakKeyB;
    }

    if (key_a < 0 or key_b < 0 or key_b > m - 1) {
        return CipherError.InvalidKeyRange;
    }

    if (gcd(key_a, m) != 1) {
        return CipherError.NotRelativelyPrime;
    }
}

/// Encrypts text with affine cipher using combined key format.
/// Combined key is split as `key_a, key_b = divmod(key, len(SYMBOLS))`.
/// Time complexity: O(n * m), Space complexity: O(n)
pub fn encryptMessage(allocator: Allocator, key: i64, message: []const u8) ![]u8 {
    const m: i64 = @intCast(SYMBOLS.len);
    const key_a = @divFloor(key, m);
    const key_b = @mod(key, m);
    try checkKeys(key_a, key_b, true);

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (message, 0..) |symbol, i| {
        const idx_opt = std.mem.indexOfScalar(u8, SYMBOLS, symbol);
        if (idx_opt == null) {
            out[i] = symbol;
            continue;
        }

        const sym_index: i64 = @intCast(idx_opt.?);
        const mapped = @mod(sym_index * key_a + key_b, m);
        out[i] = SYMBOLS[@intCast(mapped)];
    }

    return out;
}

/// Decrypts affine-cipher text with combined key format.
/// Time complexity: O(n * m), Space complexity: O(n)
pub fn decryptMessage(allocator: Allocator, key: i64, message: []const u8) ![]u8 {
    const m: i64 = @intCast(SYMBOLS.len);
    const key_a = @divFloor(key, m);
    const key_b = @mod(key, m);
    try checkKeys(key_a, key_b, false);

    const inv = try modInverse(key_a, m);

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (message, 0..) |symbol, i| {
        const idx_opt = std.mem.indexOfScalar(u8, SYMBOLS, symbol);
        if (idx_opt == null) {
            out[i] = symbol;
            continue;
        }

        const sym_index: i64 = @intCast(idx_opt.?);
        const mapped = @mod((sym_index - key_b) * inv, m);
        out[i] = SYMBOLS[@intCast(mapped)];
    }

    return out;
}

test "affine cipher: python sample" {
    const alloc = testing.allocator;
    const plain = "The affine cipher is a type of monoalphabetic substitution cipher.";
    const encrypted = try encryptMessage(alloc, 4545, plain);
    defer alloc.free(encrypted);

    try testing.expectEqualStrings(
        "VL}p MM{I}p~{HL}Gp{vp pFsH}pxMpyxIx JHL O}F{~pvuOvF{FuF{xIp~{HL}Gi",
        encrypted,
    );

    const decrypted = try decryptMessage(alloc, 4545, encrypted);
    defer alloc.free(decrypted);
    try testing.expectEqualStrings(plain, decrypted);
}

test "affine cipher: invalid keys" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.WeakKeyA, encryptMessage(alloc, 1 * @as(i64, SYMBOLS.len) + 2, "abc"));
    try testing.expectError(CipherError.WeakKeyB, encryptMessage(alloc, 2 * @as(i64, SYMBOLS.len), "abc"));
    try testing.expectError(CipherError.NotRelativelyPrime, encryptMessage(alloc, 5 * @as(i64, SYMBOLS.len) + 3, "abc"));
}

test "affine cipher: preserve unknown symbols" {
    const alloc = testing.allocator;
    const msg = "hello\nworld\t!";
    const enc = try encryptMessage(alloc, 4545, msg);
    defer alloc.free(enc);
    const dec = try decryptMessage(alloc, 4545, enc);
    defer alloc.free(dec);
    try testing.expectEqualStrings(msg, dec);
}

test "affine cipher: extreme long round-trip" {
    const alloc = testing.allocator;
    const n: usize = 12000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);
    for (0..n) |i| text[i] = SYMBOLS[i % SYMBOLS.len];

    const enc = try encryptMessage(alloc, 4545, text);
    defer alloc.free(enc);
    const dec = try decryptMessage(alloc, 4545, enc);
    defer alloc.free(dec);
    try testing.expectEqualSlices(u8, text, dec);
}
