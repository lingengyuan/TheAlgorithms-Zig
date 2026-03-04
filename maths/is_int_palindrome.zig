//! Integer Palindrome Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/is_int_palindrome.py

const std = @import("std");
const testing = std.testing;

/// Returns true when `num` is a palindromic integer in base-10.
/// Time complexity: O(d), Space complexity: O(1)
pub fn isIntPalindrome(num: i64) bool {
    if (num < 0) return false;

    const original: u128 = @intCast(num);
    var value: u128 = original;
    var reversed: u128 = 0;

    while (value > 0) {
        reversed = reversed * 10 + (value % 10);
        value /= 10;
    }

    return reversed == original;
}

test "int palindrome: python reference examples" {
    try testing.expect(!isIntPalindrome(-121));
    try testing.expect(isIntPalindrome(0));
    try testing.expect(!isIntPalindrome(10));
    try testing.expect(isIntPalindrome(11));
    try testing.expect(isIntPalindrome(101));
    try testing.expect(!isIntPalindrome(120));
}

test "int palindrome: boundary and extreme cases" {
    try testing.expect(isIntPalindrome(1_234_567_898_765_432_1));
    try testing.expect(!isIntPalindrome(std.math.maxInt(i64)));
    try testing.expect(!isIntPalindrome(std.math.minInt(i64)));
}
