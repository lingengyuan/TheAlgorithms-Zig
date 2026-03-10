//! Project Euler Problem 164: Three Consecutive Digital Sum Limit - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_164/sol1.py

const std = @import("std");
const testing = std.testing;

fn solve(
    memo: []?u64,
    digits_left: usize,
    prev1: u8,
    prev2: u8,
    first: bool,
) u64 {
    if (digits_left == 0) return 1;

    const first_index: usize = if (first) 1 else 0;
    const index = (((digits_left * 10) + prev1) * 10 + prev2) * 2 + first_index;
    if (memo[index]) |cached| return cached;

    var total: u64 = 0;
    const max_digit: u8 = 9 - prev1 - prev2;
    var current: u8 = 0;
    while (current <= max_digit) : (current += 1) {
        if (first and current == 0) continue;
        total += solve(memo, digits_left - 1, current, prev1, false);
    }

    memo[index] = total;
    return total;
}

/// Counts `n_digits`-digit numbers whose every three consecutive digits sum to at most 9.
/// Time complexity: O(n_digits * 10 * 10 * 10)
/// Space complexity: O(n_digits * 10 * 10)
pub fn solution(n_digits: usize) u64 {
    if (n_digits == 0) return 0;

    var memo_storage = [_]?u64{null} ** ((21 * 10 * 10 * 2));
    return solve(&memo_storage, n_digits, 0, 0, true);
}

test "problem 164: python reference" {
    try testing.expectEqual(@as(u64, 45), solution(2));
    try testing.expectEqual(@as(u64, 165), solution(3));
    try testing.expectEqual(@as(u64, 21838806), solution(10));
    try testing.expectEqual(@as(u64, 378158756814587), solution(20));
}

test "problem 164: edge digit lengths" {
    var memo_storage = [_]?u64{null} ** (21 * 10 * 10 * 2);
    try testing.expectEqual(@as(u64, 0), solution(0));
    try testing.expectEqual(@as(u64, 9), solution(1));
    try testing.expectEqual(@as(u64, 45), solve(&memo_storage, 2, 0, 0, true));
}
