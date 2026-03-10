//! Project Euler Problem 129: Repunit Divisibility - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_129/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn leastDivisibleRepunit(divisor: u32) u32 {
    if (divisor % 5 == 0 or divisor % 2 == 0) return 0;
    var repunit: u32 = 1;
    var repunit_index: u32 = 1;
    while (repunit != 0) {
        repunit = (10 * repunit + 1) % divisor;
        repunit_index += 1;
    }
    return repunit_index;
}

/// Returns the least n for which A(n) first exceeds `limit`.
/// Time complexity: unbounded search; practical runtime is acceptable for tested limits
/// Space complexity: O(1)
pub fn solution(limit: u32) u32 {
    var divisor = limit - 1;
    if ((divisor & 1) == 0) divisor += 1;
    while (leastDivisibleRepunit(divisor) <= limit) divisor += 2;
    return divisor;
}

test "problem 129: helper" {
    try testing.expectEqual(@as(u32, 6), leastDivisibleRepunit(7));
    try testing.expectEqual(@as(u32, 5), leastDivisibleRepunit(41));
    try testing.expectEqual(@as(u32, 34020), leastDivisibleRepunit(1234567));
}

test "problem 129: python reference" {
    try testing.expectEqual(@as(u32, 17), solution(10));
    try testing.expectEqual(@as(u32, 109), solution(100));
    try testing.expectEqual(@as(u32, 1017), solution(1000));
    try testing.expectEqual(@as(u32, 1000023), solution(1_000_000));
}
