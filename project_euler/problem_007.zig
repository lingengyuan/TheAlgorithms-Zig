//! Project Euler Problem 7: 10001st Prime - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_007/sol1.py

const std = @import("std");
const testing = std.testing;

/// Checks primality in O(sqrt(n)) using 6k +/- 1 optimization.
pub fn isPrime(number: u64) bool {
    if (number == 2 or number == 3) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: u64 = 5;
    while (i * i <= number) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) return false;
    }
    return true;
}

/// Returns the nth prime number.
/// Python-reference edge behavior: `n <= 0` returns 1.
///
/// Time complexity: O(n * sqrt(p_n))
/// Space complexity: O(1)
pub fn solution(nth: i64) u64 {
    var count: i64 = 0;
    var number: u64 = 1;

    while (count != nth and number < 3) {
        number += 1;
        if (isPrime(number)) count += 1;
    }

    while (count != nth) {
        number += 2;
        if (isPrime(number)) count += 1;
    }

    return number;
}

test "problem 007: python examples" {
    try testing.expectEqual(@as(u64, 13), solution(6));
    try testing.expectEqual(@as(u64, 2), solution(1));
    try testing.expectEqual(@as(u64, 5), solution(3));
    try testing.expectEqual(@as(u64, 71), solution(20));
    try testing.expectEqual(@as(u64, 229), solution(50));
    try testing.expectEqual(@as(u64, 541), solution(100));
}

test "problem 007: boundaries and official case" {
    try testing.expectEqual(@as(u64, 1), solution(0));
    try testing.expectEqual(@as(u64, 104743), solution(10001));

    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(27));
    try testing.expect(isPrime(2999));
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
}
