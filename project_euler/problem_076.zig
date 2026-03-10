//! Project Euler Problem 76: Counting Summations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_076/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of ways `m` can be written as a sum of at least two positive integers.
/// Time complexity: O(m^2)
/// Space complexity: O(m)
pub fn solution(allocator: std.mem.Allocator, m: usize) !u64 {
    if (m <= 1) return 0;

    var ways = try allocator.alloc(u64, m + 1);
    defer allocator.free(ways);
    @memset(ways, 0);
    ways[0] = 1;

    var part: usize = 1;
    while (part < m) : (part += 1) {
        var total = part;
        while (total <= m) : (total += 1) {
            ways[total] += ways[total - part];
        }
    }

    return ways[m];
}

test "problem 076: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 190569291), try solution(alloc, 100));
    try testing.expectEqual(@as(u64, 204225), try solution(alloc, 50));
    try testing.expectEqual(@as(u64, 5603), try solution(alloc, 30));
    try testing.expectEqual(@as(u64, 41), try solution(alloc, 10));
}

test "problem 076: small and edge values" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 6), try solution(alloc, 5));
    try testing.expectEqual(@as(u64, 2), try solution(alloc, 3));
    try testing.expectEqual(@as(u64, 1), try solution(alloc, 2));
    try testing.expectEqual(@as(u64, 0), try solution(alloc, 1));
}
