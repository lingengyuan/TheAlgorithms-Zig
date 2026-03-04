//! Count Vowels - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/count_vowels.py

const std = @import("std");
const testing = std.testing;

/// Counts vowels in a string.
/// Time complexity: O(n), Space complexity: O(1)
pub fn countVowels(s: []const u8) usize {
    var count: usize = 0;
    for (s) |char| {
        switch (char) {
            'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' => count += 1,
            else => {},
        }
    }
    return count;
}

test "count vowels: python reference examples" {
    try testing.expectEqual(@as(usize, 3), countVowels("hello world"));
    try testing.expectEqual(@as(usize, 3), countVowels("HELLO WORLD"));
    try testing.expectEqual(@as(usize, 3), countVowels("123 hello world"));
    try testing.expectEqual(@as(usize, 0), countVowels(""));
    try testing.expectEqual(@as(usize, 5), countVowels("a quick brown fox"));
    try testing.expectEqual(@as(usize, 5), countVowels("the quick BROWN fox"));
    try testing.expectEqual(@as(usize, 1), countVowels("PYTHON"));
}

test "count vowels: edge and extreme cases" {
    var long_input = [_]u8{'a'} ** 250_000;
    try testing.expectEqual(@as(usize, 250_000), countVowels(&long_input));
    try testing.expectEqual(@as(usize, 0), countVowels("bcdfg"));
}
