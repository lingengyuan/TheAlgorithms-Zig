//! Onepad Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/onepad_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const OnepadError = error{
    CipherTooShort,
    InvalidKeyValue,
    InvalidPlainValue,
};

pub const OnepadResult = struct {
    cipher: []i64,
    key: []i64,

    pub fn deinit(self: OnepadResult, allocator: Allocator) void {
        allocator.free(self.cipher);
        allocator.free(self.key);
    }
};

/// Encrypts bytes using pseudo-random key values in [1, 300].
/// Time complexity: O(n), Space complexity: O(n)
pub fn onepadEncrypt(allocator: Allocator, random: std.Random, text: []const u8) !OnepadResult {
    const cipher = try allocator.alloc(i64, text.len);
    errdefer allocator.free(cipher);

    const key = try allocator.alloc(i64, text.len);
    errdefer allocator.free(key);

    for (text, 0..) |ch, i| {
        const k = random.intRangeAtMost(i64, 1, 300);
        const c = (@as(i64, ch) + k) * k;
        key[i] = k;
        cipher[i] = c;
    }

    return OnepadResult{ .cipher = cipher, .key = key };
}

/// Decrypts Onepad cipher sequence with provided key sequence.
/// Decryption length follows `key.len`, matching Python reference behavior.
/// Time complexity: O(n), Space complexity: O(n)
pub fn onepadDecrypt(allocator: Allocator, cipher: []const i64, key: []const i64) ![]u8 {
    if (cipher.len < key.len) return OnepadError.CipherTooShort;

    const out = try allocator.alloc(u8, key.len);
    errdefer allocator.free(out);

    for (key, 0..) |k, i| {
        if (k <= 0) return OnepadError.InvalidKeyValue;
        const p = @divTrunc(cipher[i] - (k * k), k);
        if (p < 0 or p > 255) return OnepadError.InvalidPlainValue;
        out[i] = @intCast(p);
    }

    return out;
}

test "onepad: empty and edge cases" {
    const alloc = testing.allocator;

    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();

    const empty = try onepadEncrypt(alloc, rng, "");
    defer empty.deinit(alloc);
    try testing.expectEqual(@as(usize, 0), empty.cipher.len);
    try testing.expectEqual(@as(usize, 0), empty.key.len);

    const dec_empty = try onepadDecrypt(alloc, &[_]i64{}, &[_]i64{});
    defer alloc.free(dec_empty);
    try testing.expectEqual(@as(usize, 0), dec_empty.len);

    try testing.expectError(OnepadError.CipherTooShort, onepadDecrypt(alloc, &[_]i64{}, &[_]i64{35}));
}

test "onepad: deterministic round trip" {
    const alloc = testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const result = try onepadEncrypt(alloc, rng, "Hello");
    defer result.deinit(alloc);

    const plain = try onepadDecrypt(alloc, result.cipher, result.key);
    defer alloc.free(plain);

    try testing.expectEqualStrings("Hello", plain);
}

test "onepad: extreme long round trip" {
    const alloc = testing.allocator;

    const n: usize = 5000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);

    for (text, 0..) |*ch, i| {
        ch.* = switch (i % 4) {
            0 => 'A',
            1 => 'z',
            2 => '0',
            else => ' ',
        };
    }

    var prng = std.Random.DefaultPrng.init(123456789);
    const rng = prng.random();

    const result = try onepadEncrypt(alloc, rng, text);
    defer result.deinit(alloc);

    const plain = try onepadDecrypt(alloc, result.cipher, result.key);
    defer alloc.free(plain);

    try testing.expectEqualSlices(u8, text, plain);
}
