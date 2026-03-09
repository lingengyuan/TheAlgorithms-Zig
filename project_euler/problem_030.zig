//! Project Euler Problem 30: Digit Fifth Powers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_030/sol1.py

const std = @import("std");
const testing = std.testing;

const DIGIT_POW5 = [10]u32{ 0, 1, 32, 243, 1024, 3125, 7776, 16807, 32768, 59049 };

/// Returns sum of fifth powers of decimal digits of `number`.
///
/// Time complexity: O(digits)
/// Space complexity: O(1)
pub fn digitsFifthPowersSum(number: u32) u32 {
    if (number == 0) return 0;

    var n = number;
    var total: u32 = 0;

    while (n > 0) {
        total += DIGIT_POW5[n % 10];
        n /= 10;
    }

    return total;
}

/// Returns sum of all numbers equal to sum of fifth powers of their digits,
/// searching the Python-reference range [1000, 1_000_000).
///
/// Time complexity: O(1_000_000 * digits)
/// Space complexity: O(1)
pub fn solution() u64 {
    var total: u64 = 0;

    var number: u32 = 1000;
    while (number < 1_000_000) : (number += 1) {
        if (number == digitsFifthPowersSum(number)) {
            total += number;
        }
    }

    return total;
}

test "problem 030: python reference" {
    try testing.expectEqual(@as(u64, 443_839), solution());
}

test "problem 030: helper boundaries and known values" {
    try testing.expectEqual(@as(u32, 0), digitsFifthPowersSum(0));
    try testing.expectEqual(@as(u32, 1), digitsFifthPowersSum(1));
    try testing.expectEqual(@as(u32, 1300), digitsFifthPowersSum(1234));

    // Known fifth-power identities contributing to final sum.
    try testing.expectEqual(@as(u32, 4150), digitsFifthPowersSum(4150));
    try testing.expectEqual(@as(u32, 4151), digitsFifthPowersSum(4151));
    try testing.expectEqual(@as(u32, 54_748), digitsFifthPowersSum(54748));
    try testing.expectEqual(@as(u32, 92_727), digitsFifthPowersSum(92727));
    try testing.expectEqual(@as(u32, 93_084), digitsFifthPowersSum(93084));
    try testing.expectEqual(@as(u32, 194_979), digitsFifthPowersSum(194979));
}
