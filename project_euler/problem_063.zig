//! Project Euler Problem 63: Powerful Digit Counts - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_063/sol1.py

const std = @import("std");
const testing = std.testing;

fn powU128(base: u8, exponent: u32) u128 {
    var result: u128 = 1;
    var i: u32 = 0;
    while (i < exponent) : (i += 1) result *= base;
    return result;
}

fn digitCount(value: u128) u32 {
    var n = value;
    var count: u32 = 1;
    while (n >= 10) : (n /= 10) count += 1;
    return count;
}

/// Returns the count of all n-digit positive integers that are also nth powers.
/// Time complexity: O(max_base * max_power * max_power)
/// Space complexity: O(1)
pub fn solution(max_base: i32, max_power: i32) usize {
    if (max_base <= 1 or max_power <= 1) return 0;

    var total: usize = 0;
    var power: u32 = 1;
    while (power < @as(u32, @intCast(max_power))) : (power += 1) {
        var base: u8 = 1;
        while (base < @as(u8, @intCast(max_base))) : (base += 1) {
            if (digitCount(powU128(base, power)) == power) total += 1;
        }
    }
    return total;
}

test "problem 063: python reference" {
    try testing.expectEqual(@as(usize, 49), solution(10, 22));
}

test "problem 063: edge and extreme limits" {
    try testing.expectEqual(@as(usize, 0), solution(0, 0));
    try testing.expectEqual(@as(usize, 0), solution(1, 1));
    try testing.expectEqual(@as(usize, 0), solution(-1, -1));
    try testing.expectEqual(@as(usize, 9), solution(10, 2));
    try testing.expectEqual(@as(usize, 15), solution(10, 3));
}
