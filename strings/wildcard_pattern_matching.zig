//! Wildcard Pattern Matching - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/wildcard_pattern_matching.py

const std = @import("std");
const testing = std.testing;

fn idx(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Matches full string against pattern with '.' and '*' semantics.
/// '.' matches any single character.
/// '*' matches zero or more of previous token.
/// Time complexity: O(n * m), Space complexity: O(n * m)
pub fn matchPattern(allocator: std.mem.Allocator, inputString: []const u8, pattern: []const u8) !bool {
    if (pattern.len > 0 and pattern[0] == '*') return false;

    const rows = inputString.len + 1;
    const cols = pattern.len + 1;
    const dp = try allocator.alloc(bool, rows * cols);
    defer allocator.free(dp);
    @memset(dp, false);

    dp[idx(cols, 0, 0)] = true;

    for (1..cols) |j| {
        if (pattern[j - 1] == '*' and j >= 2) {
            dp[idx(cols, 0, j)] = dp[idx(cols, 0, j - 2)];
        }
    }

    for (1..rows) |i| {
        for (1..cols) |j| {
            if (pattern[j - 1] == inputString[i - 1] or pattern[j - 1] == '.') {
                dp[idx(cols, i, j)] = dp[idx(cols, i - 1, j - 1)];
            } else if (pattern[j - 1] == '*') {
                if (j >= 2) {
                    if (dp[idx(cols, i, j - 2)]) {
                        dp[idx(cols, i, j)] = true;
                    } else if (pattern[j - 2] == inputString[i - 1] or pattern[j - 2] == '.') {
                        dp[idx(cols, i, j)] = dp[idx(cols, i - 1, j)];
                    }
                }
            }
        }
    }

    return dp[idx(cols, rows - 1, cols - 1)];
}

test "wildcard pattern matching: python reference examples" {
    const alloc = testing.allocator;

    try testing.expect(try matchPattern(alloc, "aab", "c*a*b"));
    try testing.expect(!(try matchPattern(alloc, "dabc", "*abc")));
    try testing.expect(!(try matchPattern(alloc, "aaa", "aa")));
    try testing.expect(try matchPattern(alloc, "aaa", "a.a"));
    try testing.expect(!(try matchPattern(alloc, "aaab", "aa*")));
    try testing.expect(try matchPattern(alloc, "aaab", ".*"));
    try testing.expect(!(try matchPattern(alloc, "a", "bbbb")));
    try testing.expect(!(try matchPattern(alloc, "", "bbbb")));
    try testing.expect(!(try matchPattern(alloc, "a", "")));
    try testing.expect(try matchPattern(alloc, "", ""));
}

test "wildcard pattern matching: boundary cases" {
    const alloc = testing.allocator;
    try testing.expect(try matchPattern(alloc, "ab", "ab"));
    try testing.expect(try matchPattern(alloc, "ab", ".*b"));
    try testing.expect(!(try matchPattern(alloc, "ab", ".*c")));
    try testing.expect(try matchPattern(alloc, "aaa", "a*a"));
}

test "wildcard pattern matching: extreme case" {
    const alloc = testing.allocator;

    const long_input = try alloc.alloc(u8, 120_000);
    defer alloc.free(long_input);
    @memset(long_input, 'a');

    try testing.expect(try matchPattern(alloc, long_input, "a*"));
    try testing.expect(!(try matchPattern(alloc, long_input, "a*b")));
}
