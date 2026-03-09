//! Project Euler Problem 43: Sub-string Divisibility - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_043/sol1.py

const std = @import("std");
const testing = std.testing;

const divisors = [_]u16{ 2, 3, 5, 7, 11, 13, 17 };

fn digitsToInt(digits: []const u8) u64 {
    var value: u64 = 0;
    for (digits) |digit| value = value * 10 + digit;
    return value;
}

fn passesPartial(digits: []const u8) bool {
    const len = digits.len;
    if (len < 4) return true;

    const triple: u16 = @as(u16, digits[len - 3]) * 100 + @as(u16, digits[len - 2]) * 10 + digits[len - 1];
    return triple % divisors[len - 4] == 0;
}

fn search(depth: usize, used: *[10]bool, digits: *[10]u8, total: *u64) void {
    if (depth == 10) {
        total.* += digitsToInt(digits);
        return;
    }

    for (0..10) |digit| {
        if (used[digit]) continue;
        digits[depth] = @intCast(digit);
        if (!passesPartial(digits[0 .. depth + 1])) continue;

        used[digit] = true;
        search(depth + 1, used, digits, total);
        used[digit] = false;
    }
}

/// Returns true if the full pandigital tuple satisfies all substring divisibility tests.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn isSubstringDivisible(num: [10]u8) bool {
    if (num[3] % 2 != 0) return false;
    if ((num[2] + num[3] + num[4]) % 3 != 0) return false;
    if (num[5] % 5 != 0) return false;

    for (divisors[3..], 0..) |divisor, idx| {
        const value: u16 = @as(u16, num[idx + 4]) * 100 + @as(u16, num[idx + 5]) * 10 + num[idx + 6];
        if (value % divisor != 0) return false;
    }
    return true;
}

/// Returns the sum of all 0-to-9 pandigital numbers with the substring divisibility property.
///
/// Time complexity: O(10!) worst-case with pruning
/// Space complexity: O(10)
pub fn solution() u64 {
    var total: u64 = 0;
    var used = [_]bool{false} ** 10;
    var digits: [10]u8 = undefined;
    search(0, &used, &digits, &total);
    return total;
}

test "problem 043: python reference" {
    try testing.expectEqual(@as(u64, 16_695_334_890), solution());
}

test "problem 043: helper semantics and extremes" {
    try testing.expect(!isSubstringDivisible(.{ 0, 1, 2, 4, 6, 5, 7, 3, 8, 9 }));
    try testing.expect(!isSubstringDivisible(.{ 5, 1, 2, 4, 6, 0, 7, 8, 3, 9 }));
    try testing.expect(isSubstringDivisible(.{ 1, 4, 0, 6, 3, 5, 7, 2, 8, 9 }));
    try testing.expectEqual(@as(u64, 1406357289), digitsToInt(&[_]u8{ 1, 4, 0, 6, 3, 5, 7, 2, 8, 9 }));
}
