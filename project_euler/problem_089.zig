//! Project Euler Problem 89: Roman Numerals - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_089/sol1.py

const std = @import("std");
const testing = std.testing;

const roman_file = @embedFile("problem_089_roman.txt");
const roman_test_file = @embedFile("problem_089_numeralcleanup_test.txt");

pub const Problem089Error = error{InvalidNumeral};

fn symbolValue(ch: u8) Problem089Error!u32 {
    return switch (ch) {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        'M' => 1000,
        else => error.InvalidNumeral,
    };
}

/// Converts a Roman numeral string to an integer.
/// Time complexity: O(len(numerals))
/// Space complexity: O(1)
pub fn parseRomanNumerals(numerals: []const u8) Problem089Error!u32 {
    if (numerals.len == 0) return error.InvalidNumeral;

    var total_value: i32 = 0;
    var index: usize = 0;
    while (index + 1 < numerals.len) : (index += 1) {
        const current_value = try symbolValue(numerals[index]);
        const next_value = try symbolValue(numerals[index + 1]);
        if (current_value < next_value) {
            total_value -= @as(i32, @intCast(current_value));
        } else {
            total_value += @as(i32, @intCast(current_value));
        }
    }
    total_value += @as(i32, @intCast(try symbolValue(numerals[numerals.len - 1])));
    return @intCast(total_value);
}

/// Writes the minimal Roman numeral for `num` into `buffer` and returns the used slice.
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn generateRomanNumerals(buffer: *[32]u8, num: u32) []const u8 {
    var value = num;
    var index: usize = 0;
    const pairs = [_]struct { value: u32, text: []const u8 }{
        .{ .value = 1000, .text = "M" },
        .{ .value = 900, .text = "CM" },
        .{ .value = 500, .text = "D" },
        .{ .value = 400, .text = "CD" },
        .{ .value = 100, .text = "C" },
        .{ .value = 90, .text = "XC" },
        .{ .value = 50, .text = "L" },
        .{ .value = 40, .text = "XL" },
        .{ .value = 10, .text = "X" },
        .{ .value = 9, .text = "IX" },
        .{ .value = 5, .text = "V" },
        .{ .value = 4, .text = "IV" },
        .{ .value = 1, .text = "I" },
    };

    for (pairs) |pair| {
        while (value >= pair.value) : (value -= pair.value) {
            @memcpy(buffer[index .. index + pair.text.len], pair.text);
            index += pair.text.len;
        }
    }
    return buffer[0..index];
}

/// Returns the number of characters saved by rewriting each Roman numeral line in minimal form.
/// Time complexity: O(total_chars)
/// Space complexity: O(1)
pub fn solution(data: []const u8) Problem089Error!usize {
    var savings: usize = 0;
    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| {
        const value = try parseRomanNumerals(line);
        var buffer: [32]u8 = undefined;
        const shortened = generateRomanNumerals(&buffer, value);
        savings += line.len - shortened.len;
    }
    return savings;
}

test "problem 089: python reference" {
    try testing.expectEqual(@as(usize, 743), try solution(roman_file));
}

test "problem 089: helpers and cleanup sample" {
    try testing.expectEqual(@as(u32, 89), try parseRomanNumerals("LXXXIX"));
    try testing.expectEqual(@as(u32, 4), try parseRomanNumerals("IIII"));
    var buffer: [32]u8 = undefined;
    try testing.expectEqualStrings("LXXXIX", generateRomanNumerals(&buffer, 89));
    try testing.expectEqualStrings("IV", generateRomanNumerals(&buffer, 4));
    try testing.expectEqual(@as(usize, 16), try solution(roman_test_file));
}
