//! Miller-Rabin Primality Test - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/deterministic_miller_rabin.py

const std = @import("std");
const testing = std.testing;

const DETERMINISTIC_BASES = [_]u64{ 2, 325, 9375, 28178, 450775, 9_780_504, 1_795_265_022 };

fn mulMod(a: u64, b: u64, m: u64) u64 {
    return @intCast((@as(u128, a) * @as(u128, b)) % @as(u128, m));
}

fn powMod(base: u64, exponent: u64, modulus: u64) u64 {
    var result: u64 = 1;
    var b = base % modulus;
    var e = exponent;
    while (e > 0) {
        if (e & 1 == 1) result = mulMod(result, b, modulus);
        b = mulMod(b, b, modulus);
        e >>= 1;
    }
    return result;
}

fn isWitness(a: u64, d: u64, s: u32, n: u64) bool {
    var x = powMod(a, d, n);
    if (x == 1 or x == n - 1) return false;

    var r: u32 = 1;
    while (r < s) : (r += 1) {
        x = mulMod(x, x, n);
        if (x == n - 1) return false;
    }

    return true;
}

/// Deterministic Miller-Rabin primality test for 64-bit integers.
/// Time complexity: O(k * log^3 n) with fixed k = 7 bases.
pub fn isPrimeMillerRabin(n: u64) bool {
    if (n < 2) return false;

    const small_primes = [_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    for (small_primes) |p| {
        if (n == p) return true;
        if (n % p == 0) return false;
    }

    var d = n - 1;
    var s: u32 = 0;
    while (d & 1 == 0) {
        d >>= 1;
        s += 1;
    }

    for (DETERMINISTIC_BASES) |base| {
        const a = base % n;
        if (a <= 1) continue;
        if (isWitness(a, d, s, n)) return false;
    }

    return true;
}

fn isPrimeTrial(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    var i: u64 = 3;
    while (i <= n / i) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

test "miller rabin: trivial cases" {
    try testing.expect(!isPrimeMillerRabin(0));
    try testing.expect(!isPrimeMillerRabin(1));
    try testing.expect(isPrimeMillerRabin(2));
    try testing.expect(isPrimeMillerRabin(3));
    try testing.expect(!isPrimeMillerRabin(4));
}

test "miller rabin: known pairs from reference ranges" {
    try testing.expect(!isPrimeMillerRabin(561));
    try testing.expect(isPrimeMillerRabin(563));

    try testing.expect(!isPrimeMillerRabin(838_201));
    try testing.expect(isPrimeMillerRabin(838_207));

    try testing.expect(!isPrimeMillerRabin(17_316_001));
    try testing.expect(isPrimeMillerRabin(17_316_017));

    try testing.expect(!isPrimeMillerRabin(3_078_386_641));
    try testing.expect(isPrimeMillerRabin(3_078_386_653));
}

test "miller rabin: matches trial division on small range" {
    var n: u64 = 2;
    while (n <= 50_000) : (n += 1) {
        try testing.expectEqual(isPrimeTrial(n), isPrimeMillerRabin(n));
    }
}

test "miller rabin: 64-bit edge values" {
    try testing.expect(!isPrimeMillerRabin(18_446_744_073_709_551_556));
    try testing.expect(isPrimeMillerRabin(18_446_744_073_709_551_557));
}
