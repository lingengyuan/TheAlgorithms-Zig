//! Project Euler Problem 190: Maximising a Weighted Product - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_190/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns sum floor(P_m) for m from 2 to n.
/// Time complexity: O(n^2)
/// Space complexity: O(1)
pub fn solution(n: u32) u64 {
    var total: u64 = 0;
    var m: u32 = 2;
    while (m <= n) : (m += 1) {
        const x1 = 2.0 / (@as(f64, @floatFromInt(m)) + 1.0);
        var p: f64 = 1.0;
        var i: u32 = 1;
        while (i <= m) : (i += 1) {
            const xi = @as(f64, @floatFromInt(i)) * x1;
            p *= std.math.pow(f64, xi, @as(f64, @floatFromInt(i)));
        }
        total += @intFromFloat(p);
    }
    return total;
}

test "problem 190: python reference" {
    try testing.expectEqual(@as(u64, 1), solution(2));
    try testing.expectEqual(@as(u64, 10), solution(5));
    try testing.expectEqual(@as(u64, 5111), solution(10));
    try testing.expectEqual(@as(u64, 371048281), solution(15));
}
