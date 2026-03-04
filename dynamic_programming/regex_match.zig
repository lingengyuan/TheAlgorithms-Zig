//! Regular Expression Matching ('.' and '*') - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/regex_match.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const RegexMatchError = error{
    Overflow,
};

fn cellIndex(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Returns whether `text` matches regex-like `pattern` where:
/// - '.' matches any single character
/// - '*' matches zero or more of the preceding element
/// Time complexity: O(text.len * pattern.len), Space complexity: O(text.len * pattern.len)
pub fn regexMatch(
    allocator: Allocator,
    text: []const u8,
    pattern: []const u8,
) (RegexMatchError || Allocator.Error)!bool {
    const rows_plus = @addWithOverflow(text.len, @as(usize, 1));
    if (rows_plus[1] != 0) return RegexMatchError.Overflow;
    const cols_plus = @addWithOverflow(pattern.len, @as(usize, 1));
    if (cols_plus[1] != 0) return RegexMatchError.Overflow;

    const total = @mulWithOverflow(rows_plus[0], cols_plus[0]);
    if (total[1] != 0) return RegexMatchError.Overflow;

    const dp = try allocator.alloc(bool, total[0]);
    defer allocator.free(dp);
    @memset(dp, false);

    dp[cellIndex(cols_plus[0], 0, 0)] = true;

    for (1..cols_plus[0]) |j| {
        if (pattern[j - 1] == '*' and j >= 2) {
            dp[cellIndex(cols_plus[0], 0, j)] = dp[cellIndex(cols_plus[0], 0, j - 2)];
        }
    }

    for (1..rows_plus[0]) |i| {
        for (1..cols_plus[0]) |j| {
            const pch = pattern[j - 1];
            const idx = cellIndex(cols_plus[0], i, j);

            if (pch == '.' or pch == text[i - 1]) {
                dp[idx] = dp[cellIndex(cols_plus[0], i - 1, j - 1)];
                continue;
            }

            if (pch == '*' and j >= 2) {
                var value = dp[cellIndex(cols_plus[0], i, j - 2)];
                const prev = pattern[j - 2];
                if (prev == '.' or prev == text[i - 1]) {
                    value = value or dp[cellIndex(cols_plus[0], i - 1, j)];
                }
                dp[idx] = value;
                continue;
            }

            dp[idx] = false;
        }
    }

    return dp[cellIndex(cols_plus[0], text.len, pattern.len)];
}

test "regex match: python doctest equivalents" {
    try testing.expect(try regexMatch(testing.allocator, "abc", "a.c"));
    try testing.expect(try regexMatch(testing.allocator, "abc", "af*.c"));
    try testing.expect(try regexMatch(testing.allocator, "abc", "a.c*"));
    try testing.expect(!(try regexMatch(testing.allocator, "abc", "a.c*d")));
    try testing.expect(try regexMatch(testing.allocator, "aa", ".*"));
}

test "regex match: boundary cases" {
    try testing.expect(try regexMatch(testing.allocator, "", ""));
    try testing.expect(!(try regexMatch(testing.allocator, "aa", "a")));
    try testing.expect(!(try regexMatch(testing.allocator, "", "a")));
    try testing.expect(!(try regexMatch(testing.allocator, "a", "")));
    try testing.expect(!(try regexMatch(testing.allocator, "abc", "*abc")));
}

test "regex match: extreme repetitive pattern and text" {
    var text_buf: [2048]u8 = undefined;
    @memset(&text_buf, 'a');
    try testing.expect(try regexMatch(testing.allocator, &text_buf, "a*"));
}
