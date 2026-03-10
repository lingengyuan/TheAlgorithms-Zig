//! Project Euler Problem 57: Square Root Convergents - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_057/sol1.py

const std = @import("std");
const testing = std.testing;

fn addDigits(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    const max_len = @max(left.len, right.len);
    var carry: u8 = 0;
    for (0..max_len) |idx| {
        const digit_left: u8 = if (idx < left.len) left[idx] else 0;
        const digit_right: u8 = if (idx < right.len) right[idx] else 0;
        const sum = digit_left + digit_right + carry;
        try out.append(allocator, sum % 10);
        carry = sum / 10;
    }
    if (carry != 0) try out.append(allocator, carry);
    return try out.toOwnedSlice(allocator);
}

fn mulDigitsSmall(allocator: std.mem.Allocator, digits: []const u8, factor: u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var carry: usize = 0;
    for (digits) |digit| {
        const value = @as(usize, digit) * factor + carry;
        try out.append(allocator, @intCast(value % 10));
        carry = value / 10;
    }
    while (carry > 0) {
        try out.append(allocator, @intCast(carry % 10));
        carry /= 10;
    }
    return try out.toOwnedSlice(allocator);
}

/// Returns the number of expansions whose numerator has more digits than the denominator.
/// Time complexity: O(n * digits), Space complexity: O(digits)
pub fn solution(allocator: std.mem.Allocator, n: usize) !usize {
    var prev_numerator = try allocator.alloc(u8, 1);
    defer allocator.free(prev_numerator);
    prev_numerator[0] = 1;

    var prev_denominator = try allocator.alloc(u8, 1);
    defer allocator.free(prev_denominator);
    prev_denominator[0] = 1;

    var result: usize = 0;
    for (0..n) |_| {
        const two_denominator = try mulDigitsSmall(allocator, prev_denominator, 2);
        defer allocator.free(two_denominator);

        const numerator = try addDigits(allocator, prev_numerator, two_denominator);
        defer allocator.free(numerator);
        const denominator = try addDigits(allocator, prev_numerator, prev_denominator);
        defer allocator.free(denominator);

        if (numerator.len > denominator.len) result += 1;

        allocator.free(prev_numerator);
        allocator.free(prev_denominator);
        prev_numerator = try allocator.dupe(u8, numerator);
        prev_denominator = try allocator.dupe(u8, denominator);
    }

    return result;
}

test "problem 057: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 2), try solution(alloc, 14));
    try testing.expectEqual(@as(usize, 15), try solution(alloc, 100));
    try testing.expectEqual(@as(usize, 1508), try solution(alloc, 10_000));
}

test "problem 057: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try solution(alloc, 0));
    try testing.expectEqual(@as(usize, 0), try solution(alloc, 1));
    try testing.expectEqual(@as(usize, 1), try solution(alloc, 8));
}
