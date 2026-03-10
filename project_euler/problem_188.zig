//! Project Euler Problem 188: Hyperexponentiation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_188/sol1.py

const std = @import("std");
const testing = std.testing;

fn modExp(base: u64, exponent: u64, modulo_value: u64) u64 {
    if (modulo_value == 1) return 0;
    var result: u64 = 1 % modulo_value;
    var b = base % modulo_value;
    var e = exponent;
    while (e > 0) : (e >>= 1) {
        if ((e & 1) == 1) result = @intCast((@as(u128, result) * b) % modulo_value);
        b = @intCast((@as(u128, b) * b) % modulo_value);
    }
    return result;
}

/// Returns the last `digits` digits of base^^height.
/// Time complexity: O(height · log exponent)
/// Space complexity: O(1)
pub fn solution(base: u64, height: u32, digits: u32) u64 {
    var modulus: u64 = 1;
    for (0..digits) |_| modulus *= 10;

    var result = base;
    var level: u32 = 1;
    while (level < height) : (level += 1) result = modExp(base, result, modulus);
    return result;
}

test "problem 188: python reference" {
    try testing.expectEqual(@as(u64, 27), solution(3, 2, 8));
    try testing.expectEqual(@as(u64, 97484987), solution(3, 3, 8));
    try testing.expectEqual(@as(u64, 2547), solution(123, 456, 4));
    try testing.expectEqual(@as(u64, 95962097), solution(1777, 1855, 8));
}

test "problem 188: custom digits" {
    try testing.expectEqual(@as(u64, 27), solution(3, 2, 2));
    try testing.expectEqual(@as(u64, 3), solution(3, 1, 8));
}
