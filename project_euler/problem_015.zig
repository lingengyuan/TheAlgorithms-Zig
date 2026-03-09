//! Project Euler Problem 15: Lattice Paths - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_015/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem015Error = error{
    Overflow,
};

fn gcd(a: u128, b: u128) u128 {
    var x = a;
    var y = b;
    while (y != 0) {
        const remainder = x % y;
        x = y;
        y = remainder;
    }
    return x;
}

/// Returns number of right/down paths in an n x n grid: C(2n, n).
///
/// Time complexity: O(n * log n)
/// Space complexity: O(1)
pub fn solution(n: u32) Problem015Error!u128 {
    if (n == 0) return 1;

    var result: u128 = 1;
    var i: u128 = 1;
    const n_u128: u128 = n;

    while (i <= n_u128) : (i += 1) {
        var numerator = n_u128 + i;
        var denominator = i;

        const g1 = gcd(numerator, denominator);
        numerator /= g1;
        denominator /= g1;

        const g2 = gcd(result, denominator);
        result /= g2;
        denominator /= g2;

        if (denominator != 1) {
            result /= denominator;
        }

        const product = @mulWithOverflow(result, numerator);
        if (product[1] != 0) return Problem015Error.Overflow;
        result = product[0];
    }

    return result;
}

test "problem 015: python reference" {
    try testing.expectEqual(@as(u128, 126_410_606_437_752), try solution(25));
    try testing.expectEqual(@as(u128, 8_233_430_727_600), try solution(23));
    try testing.expectEqual(@as(u128, 137_846_528_820), try solution(20));
    try testing.expectEqual(@as(u128, 155_117_520), try solution(15));
    try testing.expectEqual(@as(u128, 2), try solution(1));
}

test "problem 015: boundaries and large values" {
    try testing.expectEqual(@as(u128, 1), try solution(0));
    try testing.expectEqual(@as(u128, 6), try solution(2));
    try testing.expectEqual(@as(u128, 112_186_277_816_662_845_432), try solution(35));
    try testing.expectEqual(@as(u128, 100_891_344_545_564_193_334_812_497_256), try solution(50));

    // Extreme overflow-prone case for fixed-width integer implementation.
    try testing.expectError(Problem015Error.Overflow, solution(200));
}
