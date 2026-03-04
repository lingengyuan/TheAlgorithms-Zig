//! Fermat Little Theorem (Modular Division) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/fermat_little_theorem.py

const std = @import("std");
const testing = std.testing;

pub const FermatError = error{
    InvalidModulus,
    NonPrimeModulus,
    NonInvertibleDivisor,
};

/// Computes `base^exponent mod modulus` with binary exponentiation.
/// Time complexity: O(log exponent), Space complexity: O(1)
pub fn binaryExponentiation(base: u64, exponent: u64, modulus: u64) FermatError!u64 {
    if (modulus == 0) return FermatError.InvalidModulus;
    if (modulus == 1) return 0;

    var result: u64 = 1 % modulus;
    var b = base % modulus;
    var e = exponent;
    while (e > 0) {
        if ((e & 1) == 1) {
            result = mulMod(result, b, modulus);
        }
        b = mulMod(b, b, modulus);
        e >>= 1;
    }
    return result;
}

/// Computes `(a / b) mod p` using Fermat's little theorem:
/// `a * (b^(p-2) mod p) mod p`, where p must be prime and b % p != 0.
/// Time complexity: O(log p), Space complexity: O(1)
pub fn fermatDivision(a: u64, b: u64, p: u64) FermatError!u64 {
    if (p < 2) return FermatError.InvalidModulus;
    if (!isPrime(p)) return FermatError.NonPrimeModulus;
    if (b % p == 0) return FermatError.NonInvertibleDivisor;

    const inv_b = try binaryExponentiation(b, p - 2, p);
    return mulMod(a % p, inv_b, p);
}

fn mulMod(a: u64, b: u64, modulus: u64) u64 {
    return @intCast((@as(u128, a) * @as(u128, b)) % @as(u128, modulus));
}

fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;

    var i: u64 = 3;
    while (i <= n / i) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

test "fermat little theorem: binary exponentiation basics" {
    try testing.expectEqual(@as(u64, 1), try binaryExponentiation(3, 4, 5));
    try testing.expectEqual(@as(u64, 4), try binaryExponentiation(11, 13, 7));
    try testing.expectEqual(@as(u64, 0), try binaryExponentiation(5, 3, 1));
}

test "fermat little theorem: modular division sample" {
    // Python sample constants: a=1_000_000_000, b=10, p=701
    try testing.expectEqual(@as(u64, 247), try fermatDivision(1_000_000_000, 10, 701));
}

test "fermat little theorem: error paths" {
    try testing.expectError(FermatError.InvalidModulus, binaryExponentiation(2, 3, 0));
    try testing.expectError(FermatError.InvalidModulus, fermatDivision(10, 3, 1));
    try testing.expectError(FermatError.NonPrimeModulus, fermatDivision(10, 3, 21));
    try testing.expectError(FermatError.NonInvertibleDivisor, fermatDivision(10, 701, 701));
}

test "fermat little theorem: large multiplicands remain safe" {
    const base: u64 = std.math.maxInt(u64) - 3;
    const modulus: u64 = std.math.maxInt(u64) - 58;
    const reduced = base % modulus;
    const expected: u64 = @intCast((@as(u128, reduced) * @as(u128, reduced)) % @as(u128, modulus));
    try testing.expectEqual(expected, try binaryExponentiation(base, 2, modulus));
}
