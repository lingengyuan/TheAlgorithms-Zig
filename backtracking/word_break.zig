//! Word Break (Backtracking) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/word_break.py

const std = @import("std");
const testing = std.testing;

fn backtrack(input: []const u8, dict: *const std.StringHashMap(void), start: usize) bool {
    if (start == input.len) return true;

    var end = start + 1;
    while (end <= input.len) : (end += 1) {
        if (dict.contains(input[start..end]) and backtrack(input, dict, end)) {
            return true;
        }
    }

    return false;
}

/// Returns whether `input` can be segmented into dictionary words.
///
/// Time complexity: exponential in input length (pure backtracking).
/// Space complexity: O(n) recursion depth.
pub fn wordBreak(
    allocator: std.mem.Allocator,
    input: []const u8,
    words: []const []const u8,
) std.mem.Allocator.Error!bool {
    var dict = std.StringHashMap(void).init(allocator);
    defer dict.deinit();

    for (words) |word| {
        try dict.put(word, {});
    }

    return backtrack(input, &dict, 0);
}

test "word break: python examples" {
    const alloc = testing.allocator;

    try testing.expect(try wordBreak(alloc, "leetcode", &[_][]const u8{ "leet", "code" }));
    try testing.expect(try wordBreak(alloc, "applepenapple", &[_][]const u8{ "apple", "pen" }));
    try testing.expect(!(try wordBreak(alloc, "catsandog", &[_][]const u8{ "cats", "dog", "sand", "and", "cat" })));
}

test "word break: edge cases" {
    const alloc = testing.allocator;

    try testing.expect(try wordBreak(alloc, "", &[_][]const u8{"any"}));
    try testing.expect(!(try wordBreak(alloc, "abc", &[_][]const u8{})));
    try testing.expect(try wordBreak(alloc, "aaaa", &[_][]const u8{ "a", "aa" }));
}

test "word break: extreme backtracking case" {
    const alloc = testing.allocator;

    // Many prefixes match, but trailing 'b' makes segmentation impossible.
    try testing.expect(!(try wordBreak(
        alloc,
        "aaaaaaaaaaaaaaab",
        &[_][]const u8{ "a", "aa", "aaa", "aaaa", "aaaaa" },
    )));
}
