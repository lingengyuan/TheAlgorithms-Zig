//! Rabin-Miller Primality Test - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/rabin_miller.py

const std = @import("std");
const testing = std.testing;

pub const RabinMillerError = error{InvalidKeySize};

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

/// Probabilistic Rabin-Miller primality test with 5 rounds.
/// Time complexity: O(k * log^3 n), Space complexity: O(1)
pub fn rabinMiller(random: std.Random, num: u64) bool {
    if (num < 2) return false;
    if (num == 2 or num == 3) return true;
    if ((num & 1) == 0) return false;

    var s = num - 1;
    var t: u32 = 0;

    while ((s & 1) == 0) {
        s >>= 1;
        t += 1;
    }

    for (0..5) |_| {
        const a = random.intRangeAtMost(u64, 2, num - 1);
        var v = powMod(a, s, num);

        if (v != 1) {
            var i: u32 = 0;
            while (v != (num - 1)) {
                if (i == t - 1) {
                    return false;
                } else {
                    i += 1;
                    v = @as(u64, @intCast((@as(u128, v) * @as(u128, v)) % num));
                }
            }
        }
    }

    return true;
}

const LOW_PRIMES = [_]u64{
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,  73,
    79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
    191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
    439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571,
    577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
    709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853,
    857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
};

/// Fast prime check using small-prime division then Rabin-Miller.
/// Time complexity: O(len(low_primes) + k * log^3 n), Space complexity: O(1)
pub fn isPrimeLowNum(random: std.Random, num: u64) bool {
    if (num < 2) return false;

    for (LOW_PRIMES) |p| {
        if (num == p) return true;
    }

    for (LOW_PRIMES) |p| {
        if (num % p == 0) return false;
    }

    return rabinMiller(random, num);
}

/// Generates a probable prime in [2^(keysize-1), 2^keysize).
/// Time complexity: probabilistic, Space complexity: O(1)
pub fn generateLargePrime(random: std.Random, keysize: u8) !u64 {
    if (keysize < 2 or keysize > 63) return RabinMillerError.InvalidKeySize;

    const lower: u64 = @as(u64, 1) << @as(u6, @intCast(keysize - 1));
    const upper: u64 = @as(u64, 1) << @as(u6, @intCast(keysize));

    while (true) {
        const num = random.intRangeLessThan(u64, lower, upper);
        if (isPrimeLowNum(random, num)) return num;
    }
}

test "rabin miller: known primes and composites" {
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();

    try testing.expect(isPrimeLowNum(rng, 2));
    try testing.expect(isPrimeLowNum(rng, 563));
    try testing.expect(!isPrimeLowNum(rng, 561));
    try testing.expect(!isPrimeLowNum(rng, 1001));
}

test "rabin miller: generate prime with bit size" {
    var prng = std.Random.DefaultPrng.init(2);
    const rng = prng.random();

    const p = try generateLargePrime(rng, 12);
    try testing.expect(p >= (1 << 11) and p < (1 << 12));
    try testing.expect(isPrimeLowNum(rng, p));
}

test "rabin miller: invalid key size" {
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();

    try testing.expectError(RabinMillerError.InvalidKeySize, generateLargePrime(rng, 1));
    try testing.expectError(RabinMillerError.InvalidKeySize, generateLargePrime(rng, 64));
}
