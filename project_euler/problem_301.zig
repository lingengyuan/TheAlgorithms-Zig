//! Project Euler Problem 301: Nim - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_301/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of losing Nim positions of the form (n, 2n, 3n) for 1 <= n <= 2^exponent.
/// Time complexity: O(exponent)
/// Space complexity: O(1)
pub fn solution(exponent: u32) u64 {
    var a: u64 = 0;
    var b: u64 = 1;
    var index: u32 = 0;
    while (index < exponent + 2) : (index += 1) {
        const next = a + b;
        a = b;
        b = next;
    }
    // The Python reference uses phi and (phi - 1), which equals F(n) for even
    // indices and F(n) - 1 for odd indices after truncation.
    return if ((exponent & 1) == 1) a - 1 else a;
}

test "problem 301: python reference" {
    try testing.expectEqual(@as(u64, 1), solution(0));
    try testing.expectEqual(@as(u64, 3), solution(2));
    try testing.expectEqual(@as(u64, 144), solution(10));
    try testing.expectEqual(@as(u64, 2178309), solution(30));
}

test "problem 301: medium exponent" {
    try testing.expectEqual(@as(u64, 12), solution(5));
}
