//! Palindrome Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/palindrome.py

const std = @import("std");
const testing = std.testing;

/// Returns true if the string is a palindrome (two-pointer approach).
/// Case-sensitive. Time complexity: O(n)
pub fn isPalindrome(s: []const u8) bool {
    if (s.len == 0) return true;
    var lo: usize = 0;
    var hi: usize = s.len - 1;
    while (lo < hi) : ({ lo += 1; hi -= 1; }) {
        if (s[lo] != s[hi]) return false;
    }
    return true;
}

test "palindrome: true cases" {
    const cases = [_][]const u8{ "MALAYALAM", "rotor", "level", "A", "BB", "amanaplanacanalpanama" };
    for (cases) |s| try testing.expect(isPalindrome(s));
}

test "palindrome: false cases" {
    const cases = [_][]const u8{ "String", "ABC", "abcdba", "AB" };
    for (cases) |s| try testing.expect(!isPalindrome(s));
}

test "palindrome: empty string" {
    try testing.expect(isPalindrome(""));
}
