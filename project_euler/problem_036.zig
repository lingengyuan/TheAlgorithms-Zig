//! Project Euler Problem 36: Double-Base Palindromes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_036/sol1.py

const std = @import("std");
const testing = std.testing;

fn isPalindromeBytes(bytes: []const u8) bool {
    var left: usize = 0;
    var right: usize = bytes.len;
    while (left < right) {
        right -= 1;
        if (bytes[left] != bytes[right]) return false;
        left += 1;
    }
    return true;
}

/// Returns true if the input integer or byte slice is a palindrome.
pub fn isPalindrome(value: anytype) bool {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .pointer => isPalindromeBytes(value),
        .array => isPalindromeBytes(value[0..]),
        .int, .comptime_int => blk: {
            var buf: [64]u8 = undefined;
            const rendered = std.fmt.bufPrint(&buf, "{}", .{value}) catch return false;
            break :blk isPalindromeBytes(rendered);
        },
        else => @compileError("Unsupported type for isPalindrome"),
    };
}

/// Returns the sum of all numbers below `n` that are palindromic in base 10 and
/// base 2.
///
/// Time complexity: O(n log n)
/// Space complexity: O(1)
pub fn solution(n: i64) u64 {
    if (n <= 1) return 0;

    var total: u64 = 0;
    var i: i64 = 1;
    while (i < n) : (i += 1) {
        if (!isPalindrome(i)) continue;

        var binary_buf: [64]u8 = undefined;
        const binary = std.fmt.bufPrint(&binary_buf, "{b}", .{i}) catch unreachable;
        if (isPalindrome(binary)) {
            total += @intCast(i);
        }
    }

    return total;
}

test "problem 036: python reference" {
    try testing.expectEqual(@as(u64, 872_187), solution(1_000_000));
    try testing.expectEqual(@as(u64, 286_602), solution(500_000));
    try testing.expectEqual(@as(u64, 286_602), solution(100_000));
    try testing.expectEqual(@as(u64, 1_772), solution(1_000));
}

test "problem 036: boundaries and palindrome helper" {
    try testing.expectEqual(@as(u64, 157), solution(100));
    try testing.expectEqual(@as(u64, 25), solution(10));
    try testing.expectEqual(@as(u64, 1), solution(2));
    try testing.expectEqual(@as(u64, 0), solution(1));
    try testing.expectEqual(@as(u64, 0), solution(0));
    try testing.expectEqual(@as(u64, 0), solution(-1));

    try testing.expect(isPalindrome(@as(i32, 909)));
    try testing.expect(!isPalindrome(@as(i32, 908)));
    try testing.expect(isPalindrome("10101"));
    try testing.expect(!isPalindrome("10111"));
}
