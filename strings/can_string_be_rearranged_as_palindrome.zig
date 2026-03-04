//! Can String Be Rearranged as Palindrome - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/can_string_be_rearranged_as_palindrome.py

const std = @import("std");
const testing = std.testing;

/// Counter-style implementation.
/// Time complexity: O(n), Space complexity: O(1) for byte alphabet.
pub fn canStringBeRearrangedAsPalindromeCounter(input_str: []const u8) bool {
    var counts = [_]u32{0} ** 256;
    for (input_str) |ch| {
        if (ch == ' ') continue;
        counts[std.ascii.toLower(ch)] += 1;
    }

    var odd_count: u32 = 0;
    for (counts) |cnt| {
        odd_count += @intCast(cnt % 2);
    }
    return odd_count < 2;
}

/// Dictionary-style implementation.
/// Time complexity: O(n), Space complexity: O(1) for byte alphabet.
pub fn canStringBeRearrangedAsPalindrome(input_str: []const u8) bool {
    if (input_str.len == 0) return true;

    var counts = [_]u32{0} ** 256;
    for (input_str) |ch| {
        if (ch == ' ') continue;
        counts[std.ascii.toLower(ch)] += 1;
    }

    var odd_char: u32 = 0;
    for (counts) |cnt| {
        if ((cnt % 2) == 1) odd_char += 1;
    }
    return !(odd_char > 1);
}

test "can rearrange palindrome: python reference examples" {
    try testing.expect(canStringBeRearrangedAsPalindromeCounter("Momo"));
    try testing.expect(!canStringBeRearrangedAsPalindromeCounter("Mother"));
    try testing.expect(!canStringBeRearrangedAsPalindromeCounter("Father"));
    try testing.expect(canStringBeRearrangedAsPalindromeCounter("A man a plan a canal Panama"));

    try testing.expect(canStringBeRearrangedAsPalindrome("Momo"));
    try testing.expect(!canStringBeRearrangedAsPalindrome("Mother"));
    try testing.expect(!canStringBeRearrangedAsPalindrome("Father"));
    try testing.expect(canStringBeRearrangedAsPalindrome("A man a plan a canal Panama"));
}

test "can rearrange palindrome: implementation consistency and extremes" {
    const cases = [_][]const u8{
        "",
        "a",
        "aa",
        "ab",
        "tact coa",
        "Never odd or even",
        "abcddcba",
        "abcddcbaz",
    };
    for (cases) |case_str| {
        try testing.expectEqual(
            canStringBeRearrangedAsPalindromeCounter(case_str),
            canStringBeRearrangedAsPalindrome(case_str),
        );
    }

    var long_pal = [_]u8{'a'} ** 50_001;
    try testing.expect(canStringBeRearrangedAsPalindrome(&long_pal));
}
