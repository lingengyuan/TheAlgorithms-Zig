//! Project Euler Problem 34: Digit Factorials - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_034/sol1.py

const std = @import("std");
const testing = std.testing;

const digit_factorial = [10]u32{ 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880 };

/// Returns the sum of factorials of decimal digits of `n`.
///
/// Time complexity: O(digits)
/// Space complexity: O(1)
pub fn sumOfDigitFactorial(n: u32) u32 {
    if (n == 0) return 1;

    var value = n;
    var total: u32 = 0;
    while (value > 0) {
        total += digit_factorial[value % 10];
        value /= 10;
    }

    return total;
}

/// Returns the sum of all numbers equal to the sum of factorials of their digits.
/// This follows the Python search bound `7 * 9! + 1`.
///
/// Time complexity: O(limit * digits)
/// Space complexity: O(1)
pub fn solution() u64 {
    const limit: u32 = 7 * digit_factorial[9] + 1;

    var total: u64 = 0;
    var value: u32 = 3;
    while (value < limit) : (value += 1) {
        if (sumOfDigitFactorial(value) == value) {
            total += value;
        }
    }

    return total;
}

test "problem 034: python reference" {
    try testing.expectEqual(@as(u64, 40_730), solution());
}

test "problem 034: helper examples and extremes" {
    try testing.expectEqual(@as(u32, 121), sumOfDigitFactorial(15));
    try testing.expectEqual(@as(u32, 1), sumOfDigitFactorial(0));
    try testing.expectEqual(@as(u32, 145), sumOfDigitFactorial(145));
    try testing.expectEqual(@as(u32, 40_585), sumOfDigitFactorial(40_585));
    try testing.expectEqual(@as(u32, 2_177_280), sumOfDigitFactorial(999_999));
}
