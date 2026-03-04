//! RSA Key Generator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/rsa_key_generator.py

const std = @import("std");
const testing = std.testing;
const rabin = @import("rabin_miller.zig");
const cryptomath = @import("cryptomath_module.zig");

pub const RsaKeyGenError = error{
    InvalidKeySize,
    ModulusTooLarge,
};

pub const KeyPair = struct {
    public_key: [2]u64, // (n, e)
    private_key: [2]u64, // (n, d)
};

fn gcd(a_init: u64, b_init: u64) u64 {
    var a = a_init;
    var b = b_init;
    while (b != 0) {
        const t = a % b;
        a = b;
        b = t;
    }
    return a;
}

fn powMod(base_init: u64, exp_init: u64, modulus: u64) u64 {
    if (modulus == 1) return 0;

    var base = base_init % modulus;
    var exp = exp_init;
    var result: u64 = 1;

    while (exp > 0) {
        if ((exp & 1) == 1) {
            result = @as(u64, @intCast((@as(u128, result) * @as(u128, base)) % modulus));
        }
        base = @as(u64, @intCast((@as(u128, base) * @as(u128, base)) % modulus));
        exp >>= 1;
    }

    return result;
}

/// Generates RSA public/private key pair.
/// key_size is in bits for prime generation range [2^(k-1), 2^k).
/// Time complexity: probabilistic due prime search.
pub fn generateKey(random: std.Random, key_size: u8) !KeyPair {
    if (key_size < 2 or key_size > 31) return RsaKeyGenError.InvalidKeySize;

    const p = try rabin.generateLargePrime(random, key_size);
    var q = try rabin.generateLargePrime(random, key_size);
    while (q == p) q = try rabin.generateLargePrime(random, key_size);

    const n128 = @as(u128, p) * @as(u128, q);
    if (n128 > std.math.maxInt(u64)) return RsaKeyGenError.ModulusTooLarge;
    const n: u64 = @intCast(n128);

    const phi: u64 = @intCast((@as(u128, p - 1) * @as(u128, q - 1)));

    const lower: u64 = @as(u64, 1) << @as(u6, @intCast(key_size - 1));
    const upper: u64 = @as(u64, 1) << @as(u6, @intCast(key_size));

    var e: u64 = 0;
    while (true) {
        e = random.intRangeLessThan(u64, lower, upper);
        if (gcd(e, phi) == 1) break;
    }

    const d_i128 = try cryptomath.findModInverse(@intCast(e), @intCast(phi));
    const d: u64 = @intCast(d_i128);

    return KeyPair{
        .public_key = .{ n, e },
        .private_key = .{ n, d },
    };
}

test "rsa key generator: small key pair consistency" {
    var prng = std.Random.DefaultPrng.init(0);
    const rng = prng.random();

    const pair = try generateKey(rng, 8);
    const n = pair.public_key[0];
    const e = pair.public_key[1];
    const d = pair.private_key[1];

    try testing.expectEqual(n, pair.private_key[0]);
    try testing.expect(e > 1 and d > 1);

    // Sanity with a small message representative.
    const m: u64 = 123;
    const c = powMod(m % n, e, n);
    const m2 = powMod(c % n, d, n);
    try testing.expectEqual(m % n, m2);
}

test "rsa key generator: invalid size" {
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();

    try testing.expectError(RsaKeyGenError.InvalidKeySize, generateKey(rng, 1));
    try testing.expectError(RsaKeyGenError.InvalidKeySize, generateKey(rng, 40));
}
