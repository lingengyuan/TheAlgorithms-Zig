//! Project Euler Problem 5: Smallest Multiple - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_005/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem005Error = error{
    NonPositiveInput,
};

fn gcd(a: u64, b: u64) u64 {
    var x = a;
    var y = b;
    while (y != 0) {
        const t = x % y;
        x = y;
        y = t;
    }
    return x;
}

fn lcm(a: u64, b: u64) u64 {
    return @divExact(a, gcd(a, b)) * b;
}

/// Returns the smallest positive number evenly divisible by all numbers from
/// 1 to `n`.
///
/// Time complexity: O(n log n)
/// Space complexity: O(1)
pub fn solution(n: i64) Problem005Error!u64 {
    if (n <= 0) return Problem005Error.NonPositiveInput;

    var acc: u64 = 1;
    var i: u64 = 2;
    while (i <= @as(u64, @intCast(n))) : (i += 1) {
        acc = lcm(acc, i);
    }
    return acc;
}

test "problem 005: python examples" {
    try testing.expectEqual(@as(u64, 2520), try solution(10));
    try testing.expectEqual(@as(u64, 360360), try solution(15));
    try testing.expectEqual(@as(u64, 232792560), try solution(22));
}

test "problem 005: boundaries and official case" {
    try testing.expectError(Problem005Error.NonPositiveInput, solution(0));
    try testing.expectError(Problem005Error.NonPositiveInput, solution(-17));

    // Python-castable float example 3.4 => 3 in Python; here caller passes int domain.
    try testing.expectEqual(@as(u64, 6), try solution(3));

    try testing.expectEqual(@as(u64, 232792560), try solution(20));
}
