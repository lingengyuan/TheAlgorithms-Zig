//! Project Euler Problem 52: Permuted Multiples - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_052/sol1.py

const std = @import("std");
const testing = std.testing;

fn digitSignature(number: u64) [10]u8 {
    var signature = [_]u8{0} ** 10;
    var current = number;
    if (current == 0) {
        signature[0] = 1;
        return signature;
    }
    while (current > 0) : (current /= 10) {
        signature[current % 10] += 1;
    }
    return signature;
}

/// Returns true when `2x..max_multiplier*x` are permutations of `x`.
///
/// Time complexity: O(max_multiplier * digits)
/// Space complexity: O(1)
pub fn arePermutedMultiples(value: u64, max_multiplier: u8) bool {
    const signature = digitSignature(value);
    var multiplier: u8 = 2;
    while (multiplier <= max_multiplier) : (multiplier += 1) {
        if (!std.meta.eql(signature, digitSignature(value * multiplier))) return false;
    }
    return true;
}

/// Returns the smallest positive integer whose first six multiples are digit permutations.
///
/// Time complexity: unbounded search; practical runtime is small
/// Space complexity: O(1)
pub fn solution() u64 {
    var value: u64 = 1;
    while (true) : (value += 1) {
        if (arePermutedMultiples(value, 6)) return value;
    }
}

test "problem 052: python reference" {
    try testing.expectEqual(@as(u64, 142_857), solution());
}

test "problem 052: helper semantics and extremes" {
    try testing.expect(arePermutedMultiples(125_874, 2));
    try testing.expect(!arePermutedMultiples(125_874, 3));
    try testing.expect(arePermutedMultiples(142_857, 6));
    try testing.expect(!arePermutedMultiples(10, 6));
}
