//! ElGamal Key Generator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/elgamal_key_generator.py

const std = @import("std");
const testing = std.testing;
const rabin = @import("rabin_miller.zig");
const cryptomath = @import("cryptomath_module.zig");

pub const ElGamalError = error{InvalidKeySize};

pub const ElGamalPublicKey = struct {
    key_size: u8,
    e1: u64,
    e2: u64,
    p: u64,
};

pub const ElGamalPrivateKey = struct {
    key_size: u8,
    d: u64,
};

pub const ElGamalKeyPair = struct {
    public_key: ElGamalPublicKey,
    private_key: ElGamalPrivateKey,
};

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

/// Finds primitive root candidate using the same checks as Python reference.
/// Time complexity: randomized.
pub fn primitiveRoot(random: std.Random, p_val: u64) u64 {
    while (true) {
        const g = random.intRangeLessThan(u64, 3, p_val);
        if (powMod(g, 2, p_val) == 1) continue;
        if (powMod(g, p_val, p_val) == 1) continue;
        return g;
    }
}

/// Generates ElGamal key pair.
/// Time complexity: probabilistic due prime generation.
pub fn generateKey(random: std.Random, key_size: u8) !ElGamalKeyPair {
    if (key_size < 2 or key_size > 31) return ElGamalError.InvalidKeySize;

    const p = try rabin.generateLargePrime(random, key_size);
    const e1 = primitiveRoot(random, p);
    const d = random.intRangeLessThan(u64, 3, p);

    const e1d_mod_p = powMod(e1, d, p);
    const e2_i128 = try cryptomath.findModInverse(@intCast(e1d_mod_p), @intCast(p));
    const e2: u64 = @intCast(e2_i128);

    return ElGamalKeyPair{
        .public_key = .{
            .key_size = key_size,
            .e1 = e1,
            .e2 = e2,
            .p = p,
        },
        .private_key = .{
            .key_size = key_size,
            .d = d,
        },
    };
}

test "elgamal keygen: consistency" {
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();

    const pair = try generateKey(rng, 10);

    try testing.expectEqual(pair.public_key.key_size, pair.private_key.key_size);
    try testing.expect(pair.public_key.e1 >= 3 and pair.public_key.e1 < pair.public_key.p);
    try testing.expect(pair.private_key.d >= 3 and pair.private_key.d < pair.public_key.p);

    // e2 is modular inverse of e1^d mod p
    const lhs = @as(u64, @intCast((@as(u128, powMod(pair.public_key.e1, pair.private_key.d, pair.public_key.p)) * @as(u128, pair.public_key.e2)) % pair.public_key.p));
    try testing.expectEqual(@as(u64, 1), lhs);
}

test "elgamal keygen: invalid key size" {
    var prng = std.Random.DefaultPrng.init(10);
    const rng = prng.random();

    try testing.expectError(ElGamalError.InvalidKeySize, generateKey(rng, 1));
    try testing.expectError(ElGamalError.InvalidKeySize, generateKey(rng, 35));
}
