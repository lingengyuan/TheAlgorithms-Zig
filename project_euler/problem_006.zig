//! Project Euler Problem 6: Sum Square Difference - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_006/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the difference between square of sums and sum of squares for first `n`
/// natural numbers.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn solution(n: i128) i128 {
    if (n <= 0) return 0;

    const sum = @divTrunc(n * (n + 1), 2);
    const sum_sq = @divTrunc(n * (n + 1) * (2 * n + 1), 6);
    return sum * sum - sum_sq;
}

test "problem 006: python examples" {
    try testing.expectEqual(@as(i128, 2640), solution(10));
    try testing.expectEqual(@as(i128, 13160), solution(15));
    try testing.expectEqual(@as(i128, 41230), solution(20));
    try testing.expectEqual(@as(i128, 1582700), solution(50));
}

test "problem 006: boundaries and official case" {
    try testing.expectEqual(@as(i128, 0), solution(-1));
    try testing.expectEqual(@as(i128, 0), solution(0));
    try testing.expectEqual(@as(i128, 0), solution(1));

    try testing.expectEqual(@as(i128, 25164150), solution(100));

    // Stress large n in i128 domain
    try testing.expectEqual(@as(i128, 250000166666416666500000), solution(1_000_000));
}
