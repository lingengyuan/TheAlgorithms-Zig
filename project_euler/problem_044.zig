//! Project Euler Problem 44: Pentagon Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_044/sol1.py

const std = @import("std");
const testing = std.testing;

fn pentagonalNum(n: u64) u64 {
    return n * (3 * n - 1) / 2;
}

/// Returns true when `n` is pentagonal.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn isPentagonal(n: u64) bool {
    const discriminant: u64 = 1 + 24 * n;
    const root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(discriminant))));
    return root * root == discriminant and (1 + root) % 6 == 0;
}

/// Returns the minimum pentagonal difference found within the Python search horizon.
///
/// Time complexity: O(limit²)
/// Space complexity: O(1)
pub fn solution(limit: u32) i64 {
    var i: u32 = 1;
    while (i < limit) : (i += 1) {
        const pent_i = pentagonalNum(i);
        var j: u32 = i;
        while (j < limit) : (j += 1) {
            const pent_j = pentagonalNum(j);
            const sum = pent_i + pent_j;
            const diff = pent_j - pent_i;
            if (isPentagonal(sum) and isPentagonal(diff)) return @intCast(diff);
        }
    }
    return -1;
}

test "problem 044: python reference" {
    try testing.expectEqual(@as(i64, 5_482_660), solution(5000));
}

test "problem 044: helper semantics and extremes" {
    try testing.expect(isPentagonal(330));
    try testing.expect(!isPentagonal(7683));
    try testing.expect(isPentagonal(2380));
    try testing.expectEqual(@as(i64, -1), solution(10));
}
