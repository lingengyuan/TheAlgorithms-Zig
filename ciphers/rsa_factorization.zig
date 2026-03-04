//! RSA Factorization - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/rsa_factorization.py

const std = @import("std");
const testing = std.testing;
const cryptomath = @import("cryptomath_module.zig");

pub const RsaFactorError = error{
    InvalidParameter,
    FactorNotFound,
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

/// Factors RSA modulus n using known private exponent d and public exponent e.
/// Returns sorted factors [p, q].
/// Time complexity: randomized, expected sublinear for RSA trapdoor setup.
pub fn rsaFactor(random: std.Random, d: u64, e: u64, n: u64) ![2]u64 {
    if (n < 4 or d == 0 or e == 0) return RsaFactorError.InvalidParameter;

    const k128 = @as(u128, d) * @as(u128, e);
    if (k128 <= 1) return RsaFactorError.InvalidParameter;
    const k_u64: u64 = @intCast(k128 - 1);

    var attempts: usize = 0;
    while (attempts < 20_000) : (attempts += 1) {
        const g = random.intRangeAtMost(u64, 2, n - 1);
        var t = k_u64;

        while (true) {
            if ((t & 1) == 0) {
                t >>= 1;
                const x = powMod(g, t, n);
                const y = gcd(if (x > 0) x - 1 else 0, n);
                if (x > 1 and y > 1 and y < n) {
                    const p = y;
                    const q = n / y;
                    return if (p < q) .{ p, q } else .{ q, p };
                }
            } else {
                break;
            }
        }
    }

    return RsaFactorError.FactorNotFound;
}

test "rsa factorization: python samples" {
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();

    try testing.expectEqualSlices(u64, &[_]u64{ 149, 173 }, &(try rsaFactor(rng, 3, 16971, 25777)));
    try testing.expectEqualSlices(u64, &[_]u64{ 113, 241 }, &(try rsaFactor(rng, 7331, 11, 27233)));
    try testing.expectEqualSlices(u64, &[_]u64{ 89, 199 }, &(try rsaFactor(rng, 4021, 13, 17711)));
}

test "rsa factorization: invalid parameters" {
    var prng = std.Random.DefaultPrng.init(2);
    const rng = prng.random();

    try testing.expectError(RsaFactorError.InvalidParameter, rsaFactor(rng, 0, 13, 17711));
    try testing.expectError(RsaFactorError.InvalidParameter, rsaFactor(rng, 4021, 0, 17711));
    try testing.expectError(RsaFactorError.InvalidParameter, rsaFactor(rng, 1, 1, 3));
}

test "rsa factorization: extreme larger semiprime" {
    const p: u64 = 65_537;
    const q: u64 = 65_539;
    const n: u64 = p * q;

    const phi: i128 = @as(i128, p - 1) * @as(i128, q - 1);
    const e: i128 = 65_537;
    const d_i128 = try cryptomath.findModInverse(e, phi);
    const d: u64 = @intCast(d_i128);

    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();

    const factors = try rsaFactor(rng, d, @intCast(e), n);
    try testing.expectEqual(@as(u64, p), factors[0]);
    try testing.expectEqual(@as(u64, q), factors[1]);
}
