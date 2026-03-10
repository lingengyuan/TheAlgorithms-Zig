//! Project Euler Problem 113: Non-Bouncy Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_113/sol1.py

const std = @import("std");
const testing = std.testing;

/// Computes the binomial coefficient C(n, r).
/// Time complexity: O(min(r, n - r))
/// Space complexity: O(1)
pub fn choose(n: u32, r: u32) u128 {
    if (r > n) return 0;
    const rr = @min(r, n - r);
    var result: u128 = 1;
    var i: u32 = 0;
    while (i < rr) : (i += 1) {
        result = result * @as(u128, n - rr + i + 1) / @as(u128, i + 1);
    }
    return result;
}

/// Returns the number of non-bouncy numbers with exactly `n` digits, matching the Python helper.
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn nonBouncyExact(n: u32) u128 {
    return choose(8 + n, n) + choose(9 + n, n) - 10;
}

/// Returns the number of non-bouncy numbers with at most `n` digits.
/// Time complexity: O(n^2)
/// Space complexity: O(1)
pub fn nonBouncyUpto(n: u32) u128 {
    var total: u128 = 0;
    for (1..n + 1) |digits| total += nonBouncyExact(@intCast(digits));
    return total;
}

/// Returns the number of non-bouncy numbers below 10^`num_digits`.
/// Time complexity: O(n^2)
/// Space complexity: O(1)
pub fn solution(num_digits: u32) u128 {
    return nonBouncyUpto(num_digits);
}

test "problem 113: combinatorics helpers" {
    try testing.expectEqual(@as(u128, 6), choose(4, 2));
    try testing.expectEqual(@as(u128, 10), choose(5, 3));
    try testing.expectEqual(@as(u128, 38760), choose(20, 6));
}

test "problem 113: python reference" {
    try testing.expectEqual(@as(u128, 7998), nonBouncyExact(6));
    try testing.expectEqual(@as(u128, 136126), nonBouncyExact(10));
    try testing.expectEqual(@as(u128, 12951), solution(6));
    try testing.expectEqual(@as(u128, 277032), solution(10));
    try testing.expectEqual(@as(u128, 51161058134250), solution(100));
}

test "problem 113: smallest digit count" {
    try testing.expectEqual(@as(u128, 9), nonBouncyExact(1));
    try testing.expectEqual(@as(u128, 9), nonBouncyUpto(1));
}
