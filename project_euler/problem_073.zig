//! Project Euler Problem 73: Counting Fractions in a Range - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_073/sol1.py

const std = @import("std");
const testing = std.testing;

fn gcd(a: u32, b: u32) u32 {
    var x = a;
    var y = b;
    while (y != 0) {
        const remainder = x % y;
        x = y;
        y = remainder;
    }
    return x;
}

/// Returns the number of reduced proper fractions between `1/3` and `1/2`
/// with denominator `<= max_d`.
/// Time complexity: O(max_d^2 log max_d)
/// Space complexity: O(1)
pub fn solution(max_d: u32) u32 {
    var fractions_number: u32 = 0;
    var d: u32 = 0;
    while (d <= max_d) : (d += 1) {
        var n_start = @divTrunc(d, 3) + 1;
        var n_step: u32 = 1;
        if (d % 2 == 0) {
            n_start += 1 - n_start % 2;
            n_step = 2;
        }
        var n = n_start;
        while (n < @divTrunc(d + 1, 2)) : (n += n_step) {
            if (gcd(n, d) == 1) fractions_number += 1;
        }
    }
    return fractions_number;
}

test "problem 073: python reference" {
    try testing.expectEqual(@as(u32, 0), solution(4));
    try testing.expectEqual(@as(u32, 1), solution(5));
    try testing.expectEqual(@as(u32, 3), solution(8));
    try testing.expectEqual(@as(u32, 7_295_372), solution(12_000));
}

test "problem 073: edge denominators" {
    try testing.expectEqual(@as(u32, 0), solution(0));
    try testing.expectEqual(@as(u32, 0), solution(1));
    try testing.expectEqual(@as(u32, 0), solution(3));
}
