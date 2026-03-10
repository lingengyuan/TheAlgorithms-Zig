//! Project Euler Problem 72: Counting Fractions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_072/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of reduced proper fractions with denominator `<= limit`.
/// Time complexity: O(limit log log limit)
/// Space complexity: O(limit)
pub fn solution(allocator: std.mem.Allocator, limit: usize) !u64 {
    if (limit < 2) return 0;

    var phi = try allocator.alloc(u64, limit + 1);
    defer allocator.free(phi);

    for (phi, 0..) |*value, index| value.* = index;

    var i: usize = 2;
    while (i <= limit) : (i += 1) {
        if (phi[i] == i) {
            var multiple = i;
            while (multiple <= limit) : (multiple += i) {
                phi[multiple] -= phi[multiple] / i;
            }
        }
    }

    var total: u64 = 0;
    i = 2;
    while (i <= limit) : (i += 1) total += phi[i];
    return total;
}

test "problem 072: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 31), try solution(alloc, 10));
    try testing.expectEqual(@as(u64, 3043), try solution(alloc, 100));
    try testing.expectEqual(@as(u64, 304191), try solution(alloc, 1_000));
    try testing.expectEqual(@as(u64, 303963552391), try solution(alloc, 1_000_000));
}

test "problem 072: small and degenerate limits" {
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(u64, 21), try solution(testing.allocator, 8));
}
