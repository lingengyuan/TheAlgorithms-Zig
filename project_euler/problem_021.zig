//! Project Euler Problem 21: Amicable Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_021/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the sum of proper divisors of n.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn sumProperDivisors(n: u64) u64 {
    if (n <= 1) return 0;

    var total: u64 = 1;
    var i: u64 = 2;

    while (i * i <= n) : (i += 1) {
        if (n % i == 0) {
            total += i;
            const pair = n / i;
            if (pair != i) {
                total += pair;
            }
        }
    }

    return total;
}

/// Returns the sum of all amicable numbers below `limit`.
///
/// Time complexity: O(limit * sqrt(limit))
/// Space complexity: O(1)
pub fn solution(limit: u64) u64 {
    if (limit <= 1) return 0;

    var total: u64 = 0;
    var i: u64 = 1;

    while (i < limit) : (i += 1) {
        const d = sumProperDivisors(i);
        if (d != i and sumProperDivisors(d) == i) {
            total += i;
        }
    }

    return total;
}

test "problem 021: python reference" {
    try testing.expectEqual(@as(u64, 31_626), solution(10_000));
    try testing.expectEqual(@as(u64, 8_442), solution(5_000));
    try testing.expectEqual(@as(u64, 504), solution(1_000));
    try testing.expectEqual(@as(u64, 0), solution(100));
    try testing.expectEqual(@as(u64, 0), solution(50));
}

test "problem 021: divisor helper and boundaries" {
    try testing.expectEqual(@as(u64, 0), sumProperDivisors(1));
    try testing.expectEqual(@as(u64, 1), sumProperDivisors(2));
    try testing.expectEqual(@as(u64, 6), sumProperDivisors(6));
    try testing.expectEqual(@as(u64, 284), sumProperDivisors(220));
    try testing.expectEqual(@as(u64, 220), sumProperDivisors(284));

    try testing.expectEqual(@as(u64, 0), solution(0));
    try testing.expectEqual(@as(u64, 0), solution(1));
    try testing.expectEqual(@as(u64, 0), solution(2));

    // Extreme boundary around first amicable pair.
    try testing.expectEqual(@as(u64, 0), solution(220));
    try testing.expectEqual(@as(u64, 220), solution(221));
    try testing.expectEqual(@as(u64, 504), solution(300));
}
