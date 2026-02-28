//! Pangram Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_pangram.py
//! A pangram contains every letter of the alphabet at least once.

const std = @import("std");
const testing = std.testing;

/// Returns true if the string contains every letter a-z (case-insensitive).
/// Time complexity: O(n)
pub fn isPangram(s: []const u8) bool {
    var seen = [_]bool{false} ** 26;
    for (s) |c| {
        if (c >= 'a' and c <= 'z') seen[c - 'a'] = true;
        if (c >= 'A' and c <= 'Z') seen[c - 'A'] = true;
    }
    for (seen) |v| {
        if (!v) return false;
    }
    return true;
}

test "pangram: true" {
    try testing.expect(isPangram("The quick brown fox jumps over the lazy dog"));
    try testing.expect(isPangram("Pack my box with five dozen liquor jugs"));
}

test "pangram: false" {
    try testing.expect(!isPangram("Hello World"));
    try testing.expect(!isPangram(""));
    try testing.expect(!isPangram("abcdefghijklmnopqrstuvwxy")); // missing z
}
