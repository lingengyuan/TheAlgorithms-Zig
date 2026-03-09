//! Project Euler Problem 45: Triangular, Pentagonal, and Hexagonal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_045/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the nth hexagonal number.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn hexagonalNum(n: u64) u64 {
    return n * (2 * n - 1);
}

fn isPentagonal(n: u64) bool {
    const discriminant: u64 = 1 + 24 * n;
    const root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(discriminant))));
    return root * root == discriminant and (1 + root) % 6 == 0;
}

/// Returns the next number that is triangular, pentagonal, and hexagonal.
///
/// Time complexity: O(search span)
/// Space complexity: O(1)
pub fn solution(start: u64) u64 {
    var n = start;
    var value = hexagonalNum(n);
    while (!isPentagonal(value)) : (n += 1) {
        value = hexagonalNum(n + 1);
    }
    return value;
}

test "problem 045: python reference" {
    try testing.expectEqual(@as(u64, 1_533_776_805), solution(144));
}

test "problem 045: helper semantics and extremes" {
    try testing.expectEqual(@as(u64, 40_755), hexagonalNum(143));
    try testing.expectEqual(@as(u64, 861), hexagonalNum(21));
    try testing.expectEqual(@as(u64, 190), hexagonalNum(10));
    try testing.expectEqual(@as(u64, 1), solution(1));
}
