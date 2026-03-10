//! Project Euler Problem 136: Singleton Difference - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_136/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the count of n < n_limit that have exactly one solution.
/// Time complexity: roughly O(n log n)
/// Space complexity: O(n_limit)
pub fn solution(allocator: std.mem.Allocator, n_limit: u32) !u32 {
    const n_sol = try allocator.alloc(u8, n_limit);
    defer allocator.free(n_sol);
    @memset(n_sol, 0);

    var delta: u32 = 1;
    while (delta < (n_limit + 1) / 4 + 1) : (delta += 1) {
        var y: i64 = @intCast(4 * delta - 1);
        while (y > delta) : (y -= 1) {
            const n = @as(u64, @intCast(y)) * (4 * delta - @as(u32, @intCast(y)));
            if (n >= n_limit) break;
            n_sol[n] += 1;
        }
    }

    var ans: u32 = 0;
    for (n_sol) |count| {
        if (count == 1) ans += 1;
    }
    return ans;
}

test "problem 136: python reference" {
    try testing.expectEqual(@as(u32, 0), try solution(testing.allocator, 3));
    try testing.expectEqual(@as(u32, 3), try solution(testing.allocator, 10));
    try testing.expectEqual(@as(u32, 25), try solution(testing.allocator, 100));
    try testing.expectEqual(@as(u32, 27), try solution(testing.allocator, 110));
}
