//! Project Euler Problem 12: Highly Divisible Triangular Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_012/sol1.py

const std = @import("std");
const testing = std.testing;

/// Counts number of divisors of n using prime factor multiplicities.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn countDivisors(value: u64) u64 {
    if (value == 0) return 0;

    var n = value;
    var divisors: u64 = 1;
    var i: u64 = 2;

    while (i * i <= n) : (i += 1) {
        var multiplicity: u64 = 0;
        while (n % i == 0) {
            n /= i;
            multiplicity += 1;
        }
        divisors *= multiplicity + 1;
    }

    if (n > 1) {
        divisors *= 2;
    }

    return divisors;
}

/// Returns the first triangular number with more than `min_divisors` divisors.
///
/// Time complexity: superlinear, dominated by repeated divisor counting
/// Space complexity: O(1)
pub fn solution(min_divisors: u64) u64 {
    var t_num: u64 = 1;
    var i: u64 = 1;

    while (true) {
        i += 1;
        t_num += i;

        if (countDivisors(t_num) > min_divisors) {
            return t_num;
        }
    }
}

test "problem 012: known divisor counts" {
    try testing.expectEqual(@as(u64, 1), countDivisors(1));
    try testing.expectEqual(@as(u64, 2), countDivisors(3));
    try testing.expectEqual(@as(u64, 4), countDivisors(6));
    try testing.expectEqual(@as(u64, 4), countDivisors(10));
    try testing.expectEqual(@as(u64, 4), countDivisors(15));
    try testing.expectEqual(@as(u64, 4), countDivisors(21));
    try testing.expectEqual(@as(u64, 6), countDivisors(28));
}

test "problem 012: python reference and boundaries" {
    try testing.expectEqual(@as(u64, 28), solution(5));
    try testing.expectEqual(@as(u64, 73920), solution(100));
    try testing.expectEqual(@as(u64, 76_576_500), solution(500));

    // Boundary for divisor counter helper
    try testing.expectEqual(@as(u64, 0), countDivisors(0));
}
