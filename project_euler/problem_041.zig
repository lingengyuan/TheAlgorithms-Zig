//! Project Euler Problem 41: Pandigital Prime - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_041/sol1.py

const std = @import("std");
const testing = std.testing;

/// Checks primality in O(sqrt(n)), matching the Python helper semantics.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn isPrime(number: u64) bool {
    if (number > 1 and number < 4) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: u64 = 5;
    while (i * i <= number) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) return false;
    }
    return true;
}

fn searchLargestPandigitalPrime(n: u8, digits: []const u8, used: []bool, depth: u8, current: u64) ?u64 {
    if (depth == n) {
        return if (isPrime(current)) current else null;
    }

    for (digits, 0..) |digit, idx| {
        if (used[idx]) continue;
        used[idx] = true;
        defer used[idx] = false;

        if (depth == n - 1 and (digit % 2 == 0 or digit == 5)) continue;

        if (searchLargestPandigitalPrime(n, digits, used, depth + 1, current * 10 + digit)) |found| {
            return found;
        }
    }
    return null;
}

/// Returns the largest pandigital prime of length `n`, or 0 if none exists.
///
/// Time complexity: O(n! * sqrt(10^n))
/// Space complexity: O(n)
pub fn solution(n: u8) u64 {
    if (n == 0 or n > 9) return 0;

    const digit_sum = @as(u32, n) * (@as(u32, n) + 1) / 2;
    if (digit_sum % 3 == 0) return 0;

    var digits: [9]u8 = undefined;
    var used = [_]bool{false} ** 9;
    for (0..n) |idx| digits[idx] = n - @as(u8, @intCast(idx));

    return searchLargestPandigitalPrime(n, digits[0..n], used[0..n], 0, 0) orelse 0;
}

test "problem 041: python reference" {
    try testing.expectEqual(@as(u64, 0), solution(2));
    try testing.expectEqual(@as(u64, 4231), solution(4));
    try testing.expectEqual(@as(u64, 7_652_413), solution(7));
}

test "problem 041: helper semantics and extremes" {
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(isPrime(563));
    try testing.expect(!isPrime(67_483));
    try testing.expectEqual(@as(u64, 0), solution(1));
    try testing.expectEqual(@as(u64, 0), solution(8));
}
