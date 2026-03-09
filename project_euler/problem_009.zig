//! Project Euler Problem 9: Special Pythagorean Triplet - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_009/sol1.py

const std = @import("std");
const testing = std.testing;

/// Brute-force search for the product a*b*c where:
/// - a < b < c
/// - a^2 + b^2 = c^2
/// - a + b + c = target_sum
///
/// Returns -1 if no triplet exists.
///
/// Time complexity: O(target_sum^3)
/// Space complexity: O(1)
pub fn solution(target_sum: i64) i64 {
    if (target_sum <= 0) return -1;

    var a: i64 = 0;
    while (a < 300) : (a += 1) {
        var b: i64 = a + 1;
        while (b < 400) : (b += 1) {
            var c: i64 = b + 1;
            while (c < 500) : (c += 1) {
                if (a + b + c == target_sum and a * a + b * b == c * c) {
                    return a * b * c;
                }
            }
        }
    }

    return -1;
}

/// Faster search using c = target_sum - a - b.
///
/// Time complexity: O(target_sum^2)
/// Space complexity: O(1)
pub fn solutionFast(target_sum: i64) i64 {
    if (target_sum <= 0) return -1;

    var a: i64 = 0;
    while (a < 300) : (a += 1) {
        var b: i64 = 0;
        while (b < 400) : (b += 1) {
            const c = target_sum - a - b;
            if (a < b and b < c and a * a + b * b == c * c) {
                return a * b * c;
            }
        }
    }

    return -1;
}

test "problem 009: python reference target" {
    try testing.expectEqual(@as(i64, 31_875_000), solution(1000));
    try testing.expectEqual(@as(i64, 31_875_000), solutionFast(1000));
}

test "problem 009: boundaries and no-solution cases" {
    try testing.expectEqual(@as(i64, -1), solution(0));
    try testing.expectEqual(@as(i64, -1), solutionFast(-1));

    try testing.expectEqual(@as(i64, -1), solution(999));
    try testing.expectEqual(@as(i64, -1), solutionFast(999));

    // Deterministic consistency between slow/fast approaches.
    try testing.expectEqual(solution(1000), solutionFast(1000));
}
