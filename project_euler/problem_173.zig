//! Project Euler Problem 173: Using Up to One Million Tiles How Many Different "Hollow" Square Laminae Can Be Formed? - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_173/sol1.py

const std = @import("std");
const testing = std.testing;

fn ceilSqrt(value: u64) u64 {
    if (value <= 1) return value;
    var root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(value))));
    while (root * root < value) : (root += 1) {}
    while (root > 0 and (root - 1) * (root - 1) >= value) : (root -= 1) {}
    return root;
}

/// Returns the number of distinct square laminae that can be formed using at most `limit` tiles.
/// Time complexity: O(limit)
/// Space complexity: O(1)
pub fn solution(limit: u64) u64 {
    var answer: u64 = 0;
    var outer_width: u64 = 3;
    while (outer_width <= limit / 4 + 1) : (outer_width += 1) {
        const outer_area = outer_width * outer_width;
        var hole_lower: u64 = 1;
        if (outer_area > limit) {
            hole_lower = @max(ceilSqrt(outer_area - limit), 1);
        }
        if (((outer_width - hole_lower) & 1) == 1) hole_lower += 1;
        if (hole_lower < outer_width) {
            answer += (outer_width - hole_lower - 2) / 2 + 1;
        }
    }
    return answer;
}

test "problem 173: python reference" {
    try testing.expectEqual(@as(u64, 41), solution(100));
    try testing.expectEqual(@as(u64, 1572729), solution(1_000_000));
}

test "problem 173: tiny limits" {
    try testing.expectEqual(@as(u64, 0), solution(7));
    try testing.expectEqual(@as(u64, 1), solution(8));
    try testing.expectEqual(@as(u64, 9), solution(32));
}
