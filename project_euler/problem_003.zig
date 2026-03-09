//! Project Euler Problem 3: Largest Prime Factor - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_003/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem003Error = error{
    NonPositiveInput,
};

/// Checks whether a number is prime using 6k +/- 1 optimization.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn isPrime(number: u64) bool {
    if (number == 2 or number == 3) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: u64 = 5;
    while (i * i <= number) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

/// Returns the largest prime factor of `n`.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn solution(n: i128) Problem003Error!u64 {
    if (n <= 0) return Problem003Error.NonPositiveInput;

    var value: u64 = @intCast(n);
    if (isPrime(value)) return value;

    while (value % 2 == 0) {
        value /= 2;
    }
    if (isPrime(value)) return value;

    var max_factor: u64 = 0;
    var i: u64 = 3;
    while (i * i <= value) : (i += 2) {
        if (value % i == 0) {
            const cofactor = value / i;
            if (isPrime(cofactor)) {
                return cofactor;
            }
            if (isPrime(i)) {
                max_factor = i;
            }
        }
    }

    return max_factor;
}

test "problem 003: python examples" {
    try testing.expectEqual(@as(u64, 29), try solution(13195));
    try testing.expectEqual(@as(u64, 5), try solution(10));
    try testing.expectEqual(@as(u64, 17), try solution(17));
}

test "problem 003: primality checks and boundary cases" {
    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(27));
    try testing.expect(isPrime(2999));
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));

    try testing.expectError(Problem003Error.NonPositiveInput, solution(0));
    try testing.expectError(Problem003Error.NonPositiveInput, solution(-17));

    // Euler official input
    try testing.expectEqual(@as(u64, 6857), try solution(600851475143));
}
