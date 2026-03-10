//! Project Euler Problem 145: How Many Reversible Numbers Are There Below One-Billion? - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_145/sol1.py

const std = @import("std");
const testing = std.testing;

const even_digits = [_]u8{ 0, 2, 4, 6, 8 };
const odd_digits = [_]u8{ 1, 3, 5, 7, 9 };

pub fn slowReversibleNumbers(remaining_length: u32, remainder: u32, digits: []u8, length: u32) u64 {
    if (remaining_length == 0) {
        if (digits[0] == 0 or digits[length - 1] == 0) return 0;

        var carry = remainder;
        var i: i32 = @intCast(length / 2);
        while (i > 0) {
            i -= 1;
            carry += digits[@intCast(i)] + digits[length - @as(u32, @intCast(i)) - 1];
            if ((carry & 1) == 0) return 0;
            carry /= 10;
        }
        return 1;
    }

    if (remaining_length == 1) {
        if ((remainder & 1) == 0) return 0;
        var result: u64 = 0;
        for (0..10) |digit| {
            digits[length / 2] = @intCast(digit);
            result += slowReversibleNumbers(0, (remainder + 2 * @as(u32, @intCast(digit))) / 10, digits, length);
        }
        return result;
    }

    var result: u64 = 0;
    for (0..10) |digit1| {
        digits[(length + remaining_length) / 2 - 1] = @intCast(digit1);
        const choices = if (((remainder + @as(u32, @intCast(digit1))) & 1) == 0) &odd_digits else &even_digits;
        for (choices) |digit2| {
            digits[(length - remaining_length) / 2] = digit2;
            result += slowReversibleNumbers(remaining_length - 2, (remainder + @as(u32, @intCast(digit1)) + digit2) / 10, digits, length);
        }
    }
    return result;
}

pub fn reversibleNumbers(remaining_length: u32, remainder: u32, digits: []u8, length: u32) u64 {
    if (((length - 1) % 4) == 0) return 0;
    return slowReversibleNumbers(remaining_length, remainder, digits, length);
}

/// Returns the number of reversible numbers below 10^max_power.
/// Time complexity: fast recursive enumeration over digit pairs
/// Space complexity: O(max_power)
pub fn solution(max_power: u32) u64 {
    var result: u64 = 0;
    var digits: [16]u8 = undefined;
    var length: u32 = 1;
    while (length <= max_power) : (length += 1) {
        @memset(digits[0..length], 0);
        result += reversibleNumbers(length, 0, digits[0..length], length);
    }
    return result;
}

test "problem 145: recursive helpers" {
    var digits1 = [_]u8{0};
    try testing.expectEqual(@as(u64, 0), slowReversibleNumbers(1, 0, &digits1, 1));
    var digits2 = [_]u8{ 0, 0 };
    try testing.expectEqual(@as(u64, 20), reversibleNumbers(2, 0, &digits2, 2));
    var digits3 = [_]u8{ 0, 0, 0 };
    try testing.expectEqual(@as(u64, 100), reversibleNumbers(3, 0, &digits3, 3));
}

test "problem 145: python reference" {
    try testing.expectEqual(@as(u64, 120), solution(3));
    try testing.expectEqual(@as(u64, 18720), solution(6));
    try testing.expectEqual(@as(u64, 68720), solution(7));
    try testing.expectEqual(@as(u64, 608720), solution(9));
}
