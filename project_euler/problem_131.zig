//! Project Euler Problem 131: Prime Cube Partnership - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_131/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn isPrime(number: u64) bool {
    if (number < 2) return false;
    if (number % 2 == 0) return number == 2;
    var divisor: u64 = 3;
    while (divisor * divisor <= number) : (divisor += 2) {
        if (number % divisor == 0) return false;
    }
    return true;
}

/// Returns the number of qualifying primes below `max_prime`.
/// Time complexity: roughly O(number of candidates · sqrt(max_prime))
/// Space complexity: O(1)
pub fn solution(max_prime: u64) u32 {
    var primes_count: u32 = 0;
    var cube_index: u64 = 1;
    var prime_candidate: u64 = 7;
    while (prime_candidate < max_prime) {
        if (isPrime(prime_candidate)) primes_count += 1;
        cube_index += 1;
        prime_candidate += 6 * cube_index;
    }
    return primes_count;
}

test "problem 131: primality helper" {
    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(1));
    try testing.expect(!isPrime(4));
}

test "problem 131: python reference" {
    try testing.expectEqual(@as(u32, 4), solution(100));
    try testing.expectEqual(@as(u32, 173), solution(1_000_000));
}
