//! Anagram Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/check_anagrams.py

const std = @import("std");
const testing = std.testing;

/// Returns true if a and b are anagrams (ignoring case, ignoring spaces).
/// Uses a fixed 128-entry ASCII frequency table.
/// Time complexity: O(n)
pub fn isAnagram(a: []const u8, b: []const u8) bool {
    var counts = [_]i32{0} ** 128;

    for (a) |c| {
        if (c == ' ') continue;
        const lower = if (c >= 'A' and c <= 'Z') c + 32 else c;
        counts[lower] += 1;
    }
    for (b) |c| {
        if (c == ' ') continue;
        const lower = if (c >= 'A' and c <= 'Z') c + 32 else c;
        counts[lower] -= 1;
    }

    for (counts) |cnt| {
        if (cnt != 0) return false;
    }
    return true;
}

test "anagram: true cases" {
    try testing.expect(isAnagram("Silent", "Listen"));
    try testing.expect(isAnagram("This is a string", "Is this a string"));
    try testing.expect(isAnagram("abc", "bca"));
}

test "anagram: false cases" {
    try testing.expect(!isAnagram("There", "Their"));
    try testing.expect(!isAnagram("hello", "world"));
}

test "anagram: empty strings" {
    try testing.expect(isAnagram("", ""));
}
