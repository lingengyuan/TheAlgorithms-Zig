//! Project Euler Problem 71: Ordered Fractions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_071/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem071Error = error{InvalidFraction, InvalidLimit};

/// Returns the numerator of the reduced proper fraction immediately to the left
/// of `numerator / denominator` among fractions with denominator `<= limit`.
/// Time complexity: O(limit)
/// Space complexity: O(1)
pub fn solution(numerator: u64, denominator: u64, limit: u64) Problem071Error!u64 {
    if (denominator == 0 or numerator >= denominator) return error.InvalidFraction;
    if (limit == 0) return error.InvalidLimit;

    var max_numerator: u64 = 0;
    var max_denominator: u64 = 1;

    var current_denominator: u64 = 1;
    while (current_denominator <= limit) : (current_denominator += 1) {
        var current_numerator = current_denominator * numerator / denominator;
        if (current_denominator % denominator == 0) current_numerator -= 1;
        if (current_numerator * max_denominator > current_denominator * max_numerator) {
            max_numerator = current_numerator;
            max_denominator = current_denominator;
        }
    }
    return max_numerator;
}

test "problem 071: python reference" {
    try testing.expectEqual(@as(u64, 428570), try solution(3, 7, 1_000_000));
    try testing.expectEqual(@as(u64, 2), try solution(3, 7, 8));
    try testing.expectEqual(@as(u64, 47), try solution(6, 7, 60));
}

test "problem 071: invalid and edge inputs" {
    try testing.expectError(error.InvalidFraction, solution(3, 3, 8));
    try testing.expectError(error.InvalidFraction, solution(1, 0, 8));
    try testing.expectError(error.InvalidLimit, solution(3, 7, 0));
    try testing.expectEqual(@as(u64, 0), try solution(1, 2, 1));
}
