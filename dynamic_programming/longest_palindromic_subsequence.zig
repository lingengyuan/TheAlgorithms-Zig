//! Longest Palindromic Subsequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_palindromic_subsequence.py

const std = @import("std");
const testing = std.testing;

/// Returns the length of the longest palindromic subsequence in `text`.
/// Time complexity: O(n^2), space complexity: O(n^2)
pub fn longestPalindromicSubsequenceLength(
    allocator: std.mem.Allocator,
    text: []const u8,
) !usize {
    const n = text.len;
    if (n == 0) return 0;

    const elem_count = @mulWithOverflow(n, n);
    if (elem_count[1] != 0) return error.Overflow;
    const dp = try allocator.alloc(usize, elem_count[0]);
    defer allocator.free(dp);
    @memset(dp, 0);

    for (0..n) |i| dp[i * n + i] = 1;

    var len: usize = 2;
    while (len <= n) : (len += 1) {
        var i: usize = 0;
        while (i + len <= n) : (i += 1) {
            const j = i + len - 1;
            if (text[i] == text[j]) {
                if (len == 2) {
                    dp[i * n + j] = 2;
                } else {
                    dp[i * n + j] = dp[(i + 1) * n + (j - 1)] + 2;
                }
            } else {
                const left = dp[(i + 1) * n + j];
                const right = dp[i * n + (j - 1)];
                dp[i * n + j] = @max(left, right);
            }
        }
    }

    return dp[n - 1];
}

test "longest palindromic subsequence: reference examples" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 4), try longestPalindromicSubsequenceLength(alloc, "bbbab"));
    try testing.expectEqual(@as(usize, 7), try longestPalindromicSubsequenceLength(alloc, "bbabcbcab"));
}

test "longest palindromic subsequence: empty and single" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try longestPalindromicSubsequenceLength(alloc, ""));
    try testing.expectEqual(@as(usize, 1), try longestPalindromicSubsequenceLength(alloc, "z"));
}

test "longest palindromic subsequence: no repeated chars" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try longestPalindromicSubsequenceLength(alloc, "abcdef"));
}

test "longest palindromic subsequence: already palindrome" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 7), try longestPalindromicSubsequenceLength(alloc, "racecarar"));
}

test "longest palindromic subsequence: extreme repeated chars" {
    const alloc = testing.allocator;
    var buf: [1024]u8 = undefined;
    @memset(&buf, 'a');
    try testing.expectEqual(@as(usize, 1024), try longestPalindromicSubsequenceLength(alloc, &buf));
}

test "longest palindromic subsequence: oversize input length returns overflow" {
    const fake_ptr: [*]const u8 = @ptrFromInt(@alignOf(u8));
    const fake_text = fake_ptr[0..std.math.maxInt(usize)];
    try testing.expectError(error.Overflow, longestPalindromicSubsequenceLength(testing.allocator, fake_text));
}
