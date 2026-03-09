//! Project Euler Problem 53: Combinatoric Selections - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_053/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns `nCr`.
///
/// Time complexity: O(min(r, n-r))
/// Space complexity: O(1)
pub fn combinations(n: u32, r: u32) u128 {
    if (r > n) return 0;
    const k = @min(r, n - r);
    var result: u128 = 1;
    var i: u32 = 1;
    while (i <= k) : (i += 1) {
        result = (result * (n - k + i)) / i;
    }
    return result;
}

/// Counts how many `nCr` values exceed `threshold` for `1 <= n <= max_n`.
///
/// Time complexity: O(max_n²)
/// Space complexity: O(1)
pub fn solution(max_n: u32, threshold: u128) u32 {
    var total: u32 = 0;
    var n: u32 = 1;
    while (n <= max_n) : (n += 1) {
        var r: u32 = 1;
        while (r <= n) : (r += 1) {
            if (combinations(n, r) > threshold) total += 1;
        }
    }
    return total;
}

test "problem 053: python reference" {
    try testing.expectEqual(@as(u32, 4_075), solution(100, 1_000_000));
}

test "problem 053: helper semantics and extremes" {
    try testing.expectEqual(@as(u128, 10), combinations(5, 3));
    try testing.expectEqual(@as(u128, 1_144_066), combinations(23, 10));
    try testing.expectEqual(@as(u128, 1), combinations(100, 0));
    try testing.expectEqual(@as(u128, 1), combinations(100, 100));
    try testing.expectEqual(@as(u32, 0), solution(10, 1_000_000));
}
