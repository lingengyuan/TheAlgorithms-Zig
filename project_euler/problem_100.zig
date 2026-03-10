//! Project Euler Problem 100: Arranged Probability - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_100/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of blue discs for the first arrangement with total discs
/// greater than `min_total` and probability 1/2 of drawing two blue discs.
/// Time complexity: O(number of Pell-style recurrence steps)
/// Space complexity: O(1)
pub fn solution(min_total: u64) u64 {
    var prev_numerator: u128 = 1;
    var prev_denominator: u128 = 0;
    var numerator: u128 = 1;
    var denominator: u128 = 1;

    const threshold = if (min_total == 0) 0 else 2 * @as(u128, min_total) - 1;
    while (numerator <= threshold) {
        prev_numerator += 2 * numerator;
        numerator += 2 * prev_numerator;

        prev_denominator += 2 * denominator;
        denominator += 2 * prev_denominator;
    }

    return @intCast((denominator + 1) / 2);
}

test "problem 100: python reference" {
    try testing.expectEqual(@as(u64, 3), solution(2));
    try testing.expectEqual(@as(u64, 15), solution(4));
    try testing.expectEqual(@as(u64, 85), solution(21));
    try testing.expectEqual(@as(u64, 756_872_327_473), solution(1_000_000_000_000));
}

test "problem 100: tiny threshold" {
    try testing.expectEqual(@as(u64, 1), solution(0));
}
