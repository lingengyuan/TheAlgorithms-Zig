//! Project Euler Problem 94: Almost Equilateral Triangles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_094/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the sum of the perimeters of all almost equilateral triangles with integral
/// area and perimeter not exceeding `max_perimeter`.
/// Time complexity: O(number of valid triangles)
/// Space complexity: O(1)
pub fn solution(max_perimeter: u64) u64 {
    var prev_value: u64 = 1;
    var value: u64 = 2;
    var perimeters_sum: u64 = 0;
    var i: u32 = 0;
    var perimeter: u64 = 0;

    while (perimeter <= max_perimeter) : (i += 1) {
        perimeters_sum += perimeter;
        prev_value += 2 * value;
        value += prev_value;
        perimeter = if (i % 2 == 0) 2 * value + 2 else 2 * value - 2;
    }

    return perimeters_sum;
}

test "problem 094: python reference" {
    try testing.expectEqual(@as(u64, 16), solution(20));
    try testing.expectEqual(@as(u64, 518_408_346), solution(1_000_000_000));
}

test "problem 094: tiny limits" {
    try testing.expectEqual(@as(u64, 0), solution(0));
    try testing.expectEqual(@as(u64, 0), solution(15));
    try testing.expectEqual(@as(u64, 16), solution(16));
}
