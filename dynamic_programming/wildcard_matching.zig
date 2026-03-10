//! Wildcard Matching - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/wildcard_matching.py

const std = @import("std");
const testing = std.testing;

fn idx(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Matches full string against wildcard pattern using `?` and `*` semantics.
/// `?` matches any single character.
/// `*` matches any sequence, including the empty sequence.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn isMatch(allocator: std.mem.Allocator, input: []const u8, pattern: []const u8) !bool {
    const rows = input.len + 1;
    const cols = pattern.len + 1;
    const dp = try allocator.alloc(bool, rows * cols);
    defer allocator.free(dp);
    @memset(dp, false);

    dp[idx(cols, 0, 0)] = true;
    for (1..cols) |j| {
        if (pattern[j - 1] == '*') dp[idx(cols, 0, j)] = dp[idx(cols, 0, j - 1)];
    }

    for (1..rows) |i| {
        for (1..cols) |j| {
            const p = pattern[j - 1];
            if (p == '?' or p == input[i - 1]) {
                dp[idx(cols, i, j)] = dp[idx(cols, i - 1, j - 1)];
            } else if (p == '*') {
                dp[idx(cols, i, j)] = dp[idx(cols, i - 1, j)] or dp[idx(cols, i, j - 1)];
            }
        }
    }

    return dp[idx(cols, rows - 1, cols - 1)];
}

test "wildcard matching: python examples" {
    const alloc = testing.allocator;
    try testing.expect(try isMatch(alloc, "", ""));
    try testing.expect(!(try isMatch(alloc, "aa", "a")));
    try testing.expect(try isMatch(alloc, "abc", "abc"));
    try testing.expect(try isMatch(alloc, "abc", "*c"));
    try testing.expect(try isMatch(alloc, "abc", "a*"));
    try testing.expect(try isMatch(alloc, "abc", "*a*"));
    try testing.expect(try isMatch(alloc, "abc", "?b?"));
    try testing.expect(try isMatch(alloc, "abc", "*?"));
    try testing.expect(!(try isMatch(alloc, "abc", "a*d")));
    try testing.expect(!(try isMatch(alloc, "abc", "a*c?")));
}

test "wildcard matching: additional python examples" {
    const alloc = testing.allocator;
    try testing.expect(!(try isMatch(alloc, "baaabab", "*****ba*****ba")));
    try testing.expect(try isMatch(alloc, "baaabab", "*****ba*****ab"));
    try testing.expect(try isMatch(alloc, "aa", "*"));
}

test "wildcard matching: extreme case" {
    const alloc = testing.allocator;
    const long_input = try alloc.alloc(u8, 5000);
    defer alloc.free(long_input);
    @memset(long_input, 'a');

    try testing.expect(try isMatch(alloc, long_input, "*"));
    try testing.expect(try isMatch(alloc, long_input, "a*"));
    try testing.expect(!(try isMatch(alloc, long_input, "a*b")));
}
