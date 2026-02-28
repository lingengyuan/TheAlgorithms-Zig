//! Word Break - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/word_break.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns true if `text` can be segmented into words from `dictionary`.
/// Time complexity: O(n * m * k), Space complexity: O(n)
/// where n = text length, m = dictionary size, k = average word length.
pub fn wordBreak(allocator: Allocator, text: []const u8, dictionary: []const []const u8) !bool {
    const n = text.len;
    const dp = try allocator.alloc(bool, n + 1);
    defer allocator.free(dp);
    @memset(dp, false);
    dp[0] = true;

    for (1..n + 1) |i| {
        for (dictionary) |word| {
            if (word.len > i) continue;
            if (!dp[i - word.len]) continue;
            if (std.mem.eql(u8, text[i - word.len .. i], word)) {
                dp[i] = true;
                break;
            }
        }
    }

    return dp[n];
}

test "word break: basic true case" {
    const alloc = testing.allocator;
    const dict = [_][]const u8{ "leet", "code" };
    try testing.expect(try wordBreak(alloc, "leetcode", &dict));
}

test "word break: basic false case" {
    const alloc = testing.allocator;
    const dict = [_][]const u8{ "cats", "dog", "sand", "and", "cat" };
    try testing.expect(!(try wordBreak(alloc, "catsandog", &dict)));
}

test "word break: repeated reuse of word" {
    const alloc = testing.allocator;
    const dict = [_][]const u8{ "apple", "pen" };
    try testing.expect(try wordBreak(alloc, "applepenapple", &dict));
}

test "word break: empty text is true" {
    const alloc = testing.allocator;
    const dict = [_][]const u8{"a"};
    try testing.expect(try wordBreak(alloc, "", &dict));
}

test "word break: empty dictionary with non-empty text is false" {
    const alloc = testing.allocator;
    const dict = [_][]const u8{};
    try testing.expect(!(try wordBreak(alloc, "abc", &dict)));
}
