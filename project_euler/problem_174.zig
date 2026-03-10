//! Project Euler Problem 174: Counting the Number of "Hollow" Square Laminae That Can Form One, Two, Three, ... Distinct Arrangements - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_174/sol1.py

const std = @import("std");
const testing = std.testing;

fn ceilSqrt(value: u64) u64 {
    if (value <= 1) return value;
    var root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(value))));
    while (root * root < value) : (root += 1) {}
    while (root > 0 and (root - 1) * (root - 1) >= value) : (root -= 1) {}
    return root;
}

/// Returns the count of tile totals `t <= t_limit` that have between 1 and `n_limit` lamina representations.
/// Time complexity: roughly O(number of laminae)
/// Space complexity: O(t_limit)
pub fn solution(allocator: std.mem.Allocator, t_limit: u64, n_limit: u32) !u64 {
    const counts = try allocator.alloc(u16, t_limit + 1);
    defer allocator.free(counts);
    @memset(counts, 0);

    var outer_width: u64 = 3;
    while (outer_width <= t_limit / 4 + 1) : (outer_width += 1) {
        const outer_area = outer_width * outer_width;
        var hole_lower: u64 = 1;
        if (outer_area > t_limit) {
            hole_lower = @max(ceilSqrt(outer_area - t_limit), 1);
        }
        hole_lower += (outer_width - hole_lower) % 2;

        var hole_width = hole_lower;
        while (hole_width < outer_width) : (hole_width += 2) {
            const tiles = outer_area - hole_width * hole_width;
            counts[tiles] += 1;
        }
    }

    var answer: u64 = 0;
    for (counts) |count| {
        if (count >= 1 and count <= n_limit) answer += 1;
    }
    return answer;
}

test "problem 174: python reference" {
    try testing.expectEqual(@as(u64, 222), try solution(testing.allocator, 1000, 5));
    try testing.expectEqual(@as(u64, 249), try solution(testing.allocator, 1000, 10));
    try testing.expectEqual(@as(u64, 2383), try solution(testing.allocator, 10_000, 10));
    try testing.expectEqual(@as(u64, 209566), try solution(testing.allocator, 1_000_000, 10));
}

test "problem 174: tiny limits" {
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 7, 10));
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 8, 10));
    try testing.expectEqual(@as(u64, 5), try solution(testing.allocator, 32, 1));
}
