//! Check Anagrams - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/check_anagrams.py

const std = @import("std");
const testing = std.testing;
const base = @import("anagram.zig");

/// Compatibility wrapper for the Python file name.
/// Returns true if two strings are anagrams after ignoring case and spaces.
/// Time complexity: O(n), Space complexity: O(1)
pub fn checkAnagrams(first: []const u8, second: []const u8) bool {
    return base.isAnagram(first, second);
}

test "check anagrams: python samples" {
    try testing.expect(checkAnagrams("Silent", "Listen"));
    try testing.expect(checkAnagrams("This is a string", "Is this a string"));
    try testing.expect(checkAnagrams("This is    a      string", "Is     this a string"));
    try testing.expect(!checkAnagrams("There", "Their"));
}

test "check anagrams: edge and extreme" {
    try testing.expect(checkAnagrams("", ""));
    try testing.expect(checkAnagrams("A gentleman", "Elegant man"));
    try testing.expect(!checkAnagrams("abc", "abcc"));
}
