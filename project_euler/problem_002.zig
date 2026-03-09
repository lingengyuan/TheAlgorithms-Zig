//! Project Euler Problem 2: Even Fibonacci Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_002/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the sum of even Fibonacci numbers less than or equal to `limit`.
///
/// Time complexity: O(k), where k is number of generated Fibonacci terms.
/// Space complexity: O(1)
pub fn solution(limit: i128) i128 {
    var i: i128 = 1;
    var j: i128 = 2;
    var total: i128 = 0;

    while (j <= limit) {
        if (@mod(j, 2) == 0) {
            total += j;
        }
        const next = i + j;
        i = j;
        j = next;
    }

    return total;
}

test "problem 002: python examples" {
    try testing.expectEqual(@as(i128, 10), solution(10));
    try testing.expectEqual(@as(i128, 10), solution(15));
    try testing.expectEqual(@as(i128, 2), solution(2));
    try testing.expectEqual(@as(i128, 0), solution(1));
    try testing.expectEqual(@as(i128, 44), solution(34));
}

test "problem 002: boundary and extreme values" {
    try testing.expectEqual(@as(i128, 0), solution(0));
    try testing.expectEqual(@as(i128, 0), solution(-100));

    // Euler official input
    try testing.expectEqual(@as(i128, 4_613_732), solution(4_000_000));

    // Large bound stress check
    try testing.expectEqual(@as(i128, 2_763_969_850_442_378), solution(4_000_000_000_000_000));
}
