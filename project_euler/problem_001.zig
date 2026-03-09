//! Project Euler Problem 1: Multiples of 3 and 5 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_001/sol2.py

const std = @import("std");
const testing = std.testing;

pub const Problem001Error = error{
    NegativeLimit,
};

fn arithmeticProgressionSum(first: i128, difference: i128, terms: i128) i128 {
    if (terms <= 0) return 0;
    return @divTrunc(terms * (2 * first + (terms - 1) * difference), 2);
}

/// Returns the sum of all multiples of 3 or 5 below `limit`.
///
/// Note: Python reference accepts broader numeric inputs; this Zig version
/// uses explicit integer domain and returns `NegativeLimit` for negatives.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn solution(limit: i128) Problem001Error!i128 {
    if (limit < 0) return Problem001Error.NegativeLimit;

    var total: i128 = 0;
    var terms = @divFloor(limit - 1, 3);
    total += arithmeticProgressionSum(3, 3, terms);

    terms = @divFloor(limit - 1, 5);
    total += arithmeticProgressionSum(5, 5, terms);

    terms = @divFloor(limit - 1, 15);
    total -= arithmeticProgressionSum(15, 15, terms);

    return total;
}

test "problem 001: python examples" {
    try testing.expectEqual(@as(i128, 0), try solution(3));
    try testing.expectEqual(@as(i128, 3), try solution(4));
    try testing.expectEqual(@as(i128, 23), try solution(10));
    try testing.expectEqual(@as(i128, 83700), try solution(600));
}

test "problem 001: boundary and extreme values" {
    try testing.expectEqual(@as(i128, 0), try solution(0));
    try testing.expectEqual(@as(i128, 0), try solution(1));
    try testing.expectError(Problem001Error.NegativeLimit, solution(-1));

    // Euler official input
    try testing.expectEqual(@as(i128, 233168), try solution(1000));

    // Large limit stress check
    try testing.expectEqual(@as(i128, 233333166668), try solution(1_000_000));
}
