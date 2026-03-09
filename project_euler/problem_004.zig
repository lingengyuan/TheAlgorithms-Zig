//! Project Euler Problem 4: Largest Palindrome Product - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_004/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem004Error = error{
    NoPalindromeInRange,
};

fn isPalindromeNumber(n: u64) bool {
    var value = n;
    var reversed: u64 = 0;

    while (value > 0) {
        reversed = reversed * 10 + (value % 10);
        value /= 10;
    }

    return reversed == n;
}

/// Returns the largest palindrome product of two 3-digit numbers that is
/// strictly less than `upper_bound`.
///
/// Time complexity: O(900^2)
/// Space complexity: O(1)
pub fn solution(upper_bound: u64) Problem004Error!u64 {
    var best: u64 = 0;

    var a: u64 = 999;
    while (a >= 100) : (a -= 1) {
        var b: u64 = 999;
        while (b >= 100) : (b -= 1) {
            const product = a * b;
            if (product >= upper_bound or product <= best) continue;

            if (isPalindromeNumber(product)) {
                best = product;
            }
        }
        if (a == 100) break;
    }

    if (best == 0) return Problem004Error.NoPalindromeInRange;
    return best;
}

test "problem 004: python examples" {
    try testing.expectEqual(@as(u64, 19591), try solution(20000));
    try testing.expectEqual(@as(u64, 29992), try solution(30000));
    try testing.expectEqual(@as(u64, 39893), try solution(40000));
}

test "problem 004: boundaries and extreme checks" {
    try testing.expectError(Problem004Error.NoPalindromeInRange, solution(10000));
    try testing.expectError(Problem004Error.NoPalindromeInRange, solution(10001));

    try testing.expectEqual(@as(u64, 10201), try solution(11000));

    // Euler search space upper edge
    try testing.expectEqual(@as(u64, 906609), try solution(998001));
    try testing.expectEqual(@as(u64, 906609), try solution(1_000_000));
}
