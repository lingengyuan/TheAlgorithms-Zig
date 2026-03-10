//! Project Euler Problem 64: Odd Period Square Roots - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_064/sol1.py

const std = @import("std");
const testing = std.testing;

fn floorSqrt(n: u32) u32 {
    return @as(u32, @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(n)))));
}

/// Returns the continued fraction period of `sqrt(n)`.
/// Perfect squares return 0.
/// Time complexity: O(period)
/// Space complexity: O(1)
pub fn continuousFractionPeriod(n: u32) u32 {
    const root = floorSqrt(n);
    if (root * root == n) return 0;

    var numerator: i64 = 0;
    var denominator: i64 = 1;
    var integer_part: i64 = root;
    var period: u32 = 0;

    while (integer_part != 2 * @as(i64, root)) {
        numerator = denominator * integer_part - numerator;
        denominator = @divTrunc(@as(i64, n) - numerator * numerator, denominator);
        integer_part = @divTrunc(@as(i64, root) + numerator, denominator);
        period += 1;
    }
    return period;
}

/// Counts non-square integers `<= n` whose continued-fraction period is odd.
/// Time complexity: O(n * average_period)
/// Space complexity: O(1)
pub fn solution(n: u32) u32 {
    var count_odd_periods: u32 = 0;
    var i: u32 = 2;
    while (i <= n) : (i += 1) {
        if (continuousFractionPeriod(i) % 2 == 1) count_odd_periods += 1;
    }
    return count_odd_periods;
}

test "problem 064: python reference" {
    try testing.expectEqual(@as(u32, 1), solution(2));
    try testing.expectEqual(@as(u32, 2), solution(5));
    try testing.expectEqual(@as(u32, 2), solution(7));
    try testing.expectEqual(@as(u32, 3), solution(11));
    try testing.expectEqual(@as(u32, 4), solution(13));
    try testing.expectEqual(@as(u32, 1322), solution(10000));
}

test "problem 064: period helper and perfect-square extremes" {
    try testing.expectEqual(@as(u32, 1), continuousFractionPeriod(2));
    try testing.expectEqual(@as(u32, 1), continuousFractionPeriod(5));
    try testing.expectEqual(@as(u32, 4), continuousFractionPeriod(7));
    try testing.expectEqual(@as(u32, 2), continuousFractionPeriod(11));
    try testing.expectEqual(@as(u32, 5), continuousFractionPeriod(13));
    try testing.expectEqual(@as(u32, 4), continuousFractionPeriod(23));
    try testing.expectEqual(@as(u32, 0), continuousFractionPeriod(4));
}
