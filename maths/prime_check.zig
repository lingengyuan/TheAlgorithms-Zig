//! Prime Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/prime_check.py

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Checks if a number is prime in O(√n).
pub fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n < 4) return true; // 2, 3
    if (n % 2 == 0 or n % 3 == 0) return false;

    // All primes > 3 are of the form 6k ± 1
    var i: u64 = 5;
    while (i * i <= n) {
        if (n % i == 0 or n % (i + 2) == 0) return false;
        i += 6;
    }
    return true;
}

test "prime check: primes" {
    const primes = [_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 563, 2999 };
    for (primes) |p| {
        try testing.expect(isPrime(p));
    }
}

test "prime check: non-primes" {
    const non_primes = [_]u64{ 0, 1, 4, 6, 9, 15, 27, 87, 105 };
    for (non_primes) |np| {
        try testing.expect(!isPrime(np));
    }
}

test "prime check: large prime" {
    try testing.expect(isPrime(104729)); // 10000th prime
}

test "prime check: large composite" {
    try testing.expect(!isPrime(67483)); // 67483 = 131 * 515 + 18... let's verify
    // 67483 is not prime per Python reference
    try testing.expect(!isPrime(67483));
}
