//! Project Euler Problem 86: Cuboid Route - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_086/sol1.py

const std = @import("std");
const testing = std.testing;

fn isPerfectSquare(value: u64) bool {
    const root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(value))));
    return root * root == value or (root + 1) * (root + 1) == value;
}

/// Returns the least M such that the number of cuboids with integral shortest route exceeds `limit`.
/// Time complexity: roughly O(answer^2)
/// Space complexity: O(1)
pub fn solution(limit: u32) u32 {
    var num_cuboids: u32 = 0;
    var max_cuboid_size: u32 = 0;
    while (num_cuboids <= limit) {
        max_cuboid_size += 1;
        var sum_shortest_sides: u32 = 2;
        while (sum_shortest_sides <= 2 * max_cuboid_size) : (sum_shortest_sides += 1) {
            if (isPerfectSquare(@as(u64, sum_shortest_sides) * sum_shortest_sides + @as(u64, max_cuboid_size) * max_cuboid_size)) {
                const lower_bound = @max(@as(u32, 1), if (sum_shortest_sides > max_cuboid_size) sum_shortest_sides - max_cuboid_size else 0);
                const upper_bound = @min(max_cuboid_size, sum_shortest_sides / 2);
                num_cuboids += upper_bound - lower_bound + 1;
            }
        }
    }
    return max_cuboid_size;
}

test "problem 086: python reference" {
    try testing.expectEqual(@as(u32, 24), solution(100));
    try testing.expectEqual(@as(u32, 72), solution(1000));
    try testing.expectEqual(@as(u32, 100), solution(2000));
    try testing.expectEqual(@as(u32, 288), solution(20000));
    try testing.expectEqual(@as(u32, 1818), solution(1_000_000));
}
