//! Project Euler Problem 58: Spiral Primes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_058/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn isPrime(number: usize) bool {
    if (number > 1 and number < 4) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: usize = 5;
    while (i <= number / i) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) return false;
    }
    return true;
}

/// Returns the side length of the square spiral for which the prime ratio on the diagonals
/// first falls below `ratio`.
/// Time complexity: roughly O(side_length * sqrt(n)) with the direct primality checks.
/// Space complexity: O(1)
pub fn solution(ratio: f64) usize {
    var side_length: usize = 3;
    var primes: usize = 3;

    while (@as(f64, @floatFromInt(primes)) / @as(f64, @floatFromInt(2 * side_length - 1)) >= ratio) {
        var value = side_length * side_length + side_length + 1;
        while (value < (side_length + 2) * (side_length + 2)) : (value += side_length + 1) {
            if (isPrime(value)) primes += 1;
        }
        side_length += 2;
    }
    return side_length;
}

test "problem 058: primality helper" {
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(27));
    try testing.expect(!isPrime(87));
    try testing.expect(isPrime(563));
    try testing.expect(isPrime(2999));
    try testing.expect(!isPrime(67483));
}

test "problem 058: python reference" {
    try testing.expectEqual(@as(usize, 11), solution(0.5));
    try testing.expectEqual(@as(usize, 309), solution(0.2));
    try testing.expectEqual(@as(usize, 11317), solution(0.111));
}

test "problem 058: edge and extreme ratios" {
    try testing.expectEqual(@as(usize, 3), solution(0.7));
    try testing.expectEqual(@as(usize, 26241), solution(0.1));
}
