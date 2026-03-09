//! Project Euler Problem 10: Summation of Primes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_010/sol1.py

const std = @import("std");
const testing = std.testing;

/// Checks primality in O(sqrt(n)) using 6k +/- 1 optimization.
pub fn isPrime(number: i64) bool {
    if (number == 2 or number == 3) return true;
    if (number < 2 or @mod(number, 2) == 0 or @mod(number, 3) == 0) return false;

    var i: i64 = 5;
    while (i * i <= number) : (i += 6) {
        if (@mod(number, i) == 0 or @mod(number, i + 2) == 0) return false;
    }
    return true;
}

/// Returns the sum of all prime numbers below `n`.
///
/// Time complexity: O(n * sqrt(n))
/// Space complexity: O(1)
pub fn solution(n: i64) i128 {
    if (n <= 2) return 0;

    var total: i128 = 2;
    var num: i64 = 3;
    while (num < n) : (num += 2) {
        if (isPrime(num)) {
            total += num;
        }
    }

    return total;
}

test "problem 010: python examples" {
    try testing.expectEqual(@as(i128, 76127), solution(1000));
    try testing.expectEqual(@as(i128, 1548136), solution(5000));
    try testing.expectEqual(@as(i128, 5736396), solution(10000));
    try testing.expectEqual(@as(i128, 10), solution(7));
}

test "problem 010: boundaries and official case" {
    try testing.expectEqual(@as(i128, 0), solution(0));
    try testing.expectEqual(@as(i128, 0), solution(2));
    try testing.expectEqual(@as(i128, 2), solution(3));

    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(27));
    try testing.expect(isPrime(2999));
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));

    try testing.expectEqual(@as(i128, 142_913_828_922), solution(2_000_000));
}
