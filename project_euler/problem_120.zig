//! Project Euler Problem 120: Square Remainders - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_120/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the sum of maximal square remainders for 3 <= a <= n.
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn solution(n: u32) u64 {
    if (n < 3) return 0;

    var total: u64 = 0;
    var a: u32 = 3;
    while (a <= n) : (a += 1) {
        total += 2 * @as(u64, a) * ((a - 1) / 2);
    }
    return total;
}

test "problem 120: python reference" {
    try testing.expectEqual(@as(u64, 300), solution(10));
    try testing.expectEqual(@as(u64, 330750), solution(100));
    try testing.expectEqual(@as(u64, 333082500), solution(1000));
}

test "problem 120: small limits" {
    try testing.expectEqual(@as(u64, 0), solution(2));
    try testing.expectEqual(@as(u64, 100), solution(7));
}
