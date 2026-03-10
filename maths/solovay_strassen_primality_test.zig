//! Solovay-Strassen Primality Test - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/solovay_strassen_primality_test.py

const std = @import("std");
const testing = std.testing;

fn addMod(a: u128, b: u128, modulus: u128) u128 {
    const left = a % modulus;
    const right = b % modulus;
    if (left >= modulus - right) return left - (modulus - right);
    return left + right;
}

fn mulMod(a: u128, b: u128, modulus: u128) u128 {
    var result: u128 = 0;
    var left = a % modulus;
    var right = b;

    while (right > 0) {
        if (right & 1 == 1) result = addMod(result, left, modulus);
        left = addMod(left, left, modulus);
        right >>= 1;
    }

    return result;
}

fn powMod(base: u128, exponent: u128, modulus: u128) u128 {
    var result: u128 = 1 % modulus;
    var current = base % modulus;
    var power = exponent;

    while (power > 0) {
        if (power & 1 == 1) result = mulMod(result, current, modulus);
        current = mulMod(current, current, modulus);
        power >>= 1;
    }

    return result;
}

/// Computes the Jacobi symbol `(a / n)`.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn jacobiSymbol(a: i128, n: i128) i8 {
    if (n <= 0 or @mod(n, 2) == 0) return 0;
    if (a == 0) return 0;
    if (a == 1) return 1;

    var aa = @mod(a, n);
    var nn = n;
    var t: i8 = 1;

    while (aa != 0) {
        while (@mod(aa, 2) == 0) {
            aa = @divTrunc(aa, 2);
            const r = @mod(nn, 8);
            if (r == 3 or r == 5) t = -t;
        }

        const temp = aa;
        aa = nn;
        nn = temp;

        if (@mod(aa, 4) == 3 and @mod(nn, 4) == 3) t = -t;
        aa = @mod(aa, nn);
    }

    return if (nn == 1) t else 0;
}

/// Returns `true` if `number` is probably prime under the Solovay-Strassen test.
/// Time complexity: O(iterations * log^3 n), Space complexity: O(1)
pub fn solovayStrassen(number: u128, iterations: u32, random: std.Random) bool {
    if (number <= 1) return false;
    if (number <= 3) return true;
    if (number % 2 == 0) return false;

    var rng = random;
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const a = rng.uintLessThanBiased(u128, number - 3) + 2;
        const x = jacobiSymbol(@intCast(a), @intCast(number));
        const y = powMod(a, (number - 1) / 2, number);

        if (x == 0 or y != @mod(@as(i128, x), @as(i128, @intCast(number)))) {
            return false;
        }
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

test "solovay strassen: jacobi symbol reference cases" {
    try testing.expectEqual(@as(i8, -1), jacobiSymbol(2, 13));
    try testing.expectEqual(@as(i8, 1), jacobiSymbol(5, 19));
    try testing.expectEqual(@as(i8, 0), jacobiSymbol(7, 14));
    try testing.expectEqual(@as(i8, 1), jacobiSymbol(1, 99));
}

test "solovay strassen: python reference examples with deterministic seed" {
    var prng1 = std.Random.DefaultPrng.init(10);
    try testing.expect(solovayStrassen(13, 5, prng1.random()));

    var prng2 = std.Random.DefaultPrng.init(10);
    try testing.expect(!solovayStrassen(9, 10, prng2.random()));

    var prng3 = std.Random.DefaultPrng.init(10);
    try testing.expect(solovayStrassen(17, 15, prng3.random()));
}

test "solovay strassen: trivial and extreme cases" {
    var prng = std.Random.DefaultPrng.init(42);
    try testing.expect(!solovayStrassen(0, 5, prng.random()));
    try testing.expect(!solovayStrassen(1, 5, prng.random()));
    try testing.expect(solovayStrassen(2, 5, prng.random()));
    try testing.expect(solovayStrassen(3, 5, prng.random()));
    try testing.expect(!solovayStrassen(4, 5, prng.random()));
    try testing.expect(!solovayStrassen(561, 12, prng.random()));
}

test "solovay strassen: agrees with trial primality on a dense small range" {
    var n: u64 = 2;
    while (n <= 500) : (n += 1) {
        var prng = std.Random.DefaultPrng.init(n * 17 + 3);
        try testing.expectEqual(isPrimeTrial(n), solovayStrassen(n, 10, prng.random()));
    }
}
