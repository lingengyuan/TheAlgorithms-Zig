//! Project Euler Problem 42: Coded Triangle Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_042/solution42.py

const std = @import("std");
const testing = std.testing;

const words_file = @embedFile("problem_042_words.txt");

/// Returns the alphabetical value of an uppercase ASCII word.
///
/// Time complexity: O(len(word))
/// Space complexity: O(1)
pub fn wordValue(word: []const u8) u32 {
    var total: u32 = 0;
    for (word) |ch| {
        if (ch >= 'A' and ch <= 'Z') total += ch - 'A' + 1;
    }
    return total;
}

/// Returns true when `n` is a triangular number.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn isTriangleNumber(n: u32) bool {
    const discriminant: u64 = 1 + 8 * @as(u64, n);
    const root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(discriminant))));
    return root * root == discriminant and (root & 1) == 1;
}

/// Counts triangle words from the Python words-file format (`"A","B",...`).
///
/// Time complexity: O(len(data))
/// Space complexity: O(1)
pub fn countTriangleWords(data: []const u8) usize {
    var count: usize = 0;
    var current: u32 = 0;

    for (data) |ch| {
        if (ch >= 'A' and ch <= 'Z') {
            current += ch - 'A' + 1;
        } else if (current != 0) {
            if (isTriangleNumber(current)) count += 1;
            current = 0;
        }
    }

    if (current != 0 and isTriangleNumber(current)) count += 1;
    return count;
}

/// Returns the number of triangle words in the bundled dataset.
pub fn solution() usize {
    return countTriangleWords(words_file);
}

test "problem 042: python reference" {
    try testing.expectEqual(@as(usize, 162), solution());
}

test "problem 042: helpers and extremes" {
    try testing.expectEqual(@as(u32, 55), wordValue("SKY"));
    try testing.expectEqual(@as(u32, 1), wordValue("A"));
    try testing.expect(isTriangleNumber(55));
    try testing.expect(isTriangleNumber(1));
    try testing.expect(!isTriangleNumber(52));
    try testing.expectEqual(@as(usize, 1), countTriangleWords("\"SKY\",\"HELLO\""));
    try testing.expectEqual(@as(usize, 0), countTriangleWords(""));
}
