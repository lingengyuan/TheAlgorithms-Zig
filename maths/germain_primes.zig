//! Germain Primes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/germain_primes.py

const std = @import("std");
const testing = std.testing;
const prime_check = @import("prime_check.zig");

pub const GermainPrimeError = error{InvalidInput};

/// Checks whether a number is a Sophie Germain prime.
/// Time complexity: O(sqrt(n)), Space complexity: O(1)
pub fn isGermainPrime(number: i64) GermainPrimeError!bool {
    if (number < 1) return error.InvalidInput;
    const n: u64 = @intCast(number);
    return prime_check.isPrime(n) and prime_check.isPrime(2 * n + 1);
}

/// Checks whether a number is a safe prime.
/// Time complexity: O(sqrt(n)), Space complexity: O(1)
pub fn isSafePrime(number: i64) GermainPrimeError!bool {
    if (number < 1) return error.InvalidInput;
    if (@rem(number - 1, 2) != 0) return false;
    const n: u64 = @intCast(number);
    return prime_check.isPrime(n) and prime_check.isPrime(@intCast(@divTrunc(number - 1, 2)));
}

test "germain primes: python reference examples" {
    try testing.expect(try isGermainPrime(3));
    try testing.expect(try isGermainPrime(11));
    try testing.expect(!(try isGermainPrime(4)));
    try testing.expect(try isGermainPrime(23));
    try testing.expect(!(try isGermainPrime(13)));

    try testing.expect(try isSafePrime(5));
    try testing.expect(try isSafePrime(11));
    try testing.expect(!(try isSafePrime(1)));
    try testing.expect(!(try isSafePrime(2)));
    try testing.expect(try isSafePrime(47));
}

test "germain primes: edge cases" {
    try testing.expectError(error.InvalidInput, isGermainPrime(0));
    try testing.expectError(error.InvalidInput, isSafePrime(0));
}
