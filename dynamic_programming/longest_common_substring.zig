//! Longest Common Substring - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_common_substring.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LongestCommonSubstringError = error{
    Overflow,
};

/// Returns a longest common contiguous substring between `text1` and `text2`.
/// If multiple answers exist, returns the first max-length one discovered
/// by the DP scan order, matching the Python reference behavior.
/// Time complexity: O(m * n), Space complexity: O(n)
pub fn longestCommonSubstring(
    allocator: Allocator,
    text1: []const u8,
    text2: []const u8,
) (LongestCommonSubstringError || Allocator.Error)![]const u8 {
    if (text1.len == 0 or text2.len == 0) return "";

    const cols = @addWithOverflow(text2.len, @as(usize, 1));
    if (cols[1] != 0) return LongestCommonSubstringError.Overflow;

    const dp = try allocator.alloc(usize, cols[0]);
    defer allocator.free(dp);
    @memset(dp, 0);

    var end_pos: usize = 0;
    var max_len: usize = 0;

    var i: usize = 1;
    while (i <= text1.len) : (i += 1) {
        var prev_diag: usize = 0;

        var j: usize = 1;
        while (j <= text2.len) : (j += 1) {
            const current_up = dp[j];
            if (text1[i - 1] == text2[j - 1]) {
                const next = @addWithOverflow(prev_diag, @as(usize, 1));
                if (next[1] != 0) return LongestCommonSubstringError.Overflow;
                dp[j] = next[0];
                if (dp[j] > max_len) {
                    max_len = dp[j];
                    end_pos = i;
                }
            } else {
                dp[j] = 0;
            }
            prev_diag = current_up;
        }
    }

    return text1[end_pos - max_len .. end_pos];
}

test "longest common substring: python examples" {
    try testing.expectEqualStrings("", try longestCommonSubstring(testing.allocator, "", ""));
    try testing.expectEqualStrings("", try longestCommonSubstring(testing.allocator, "a", ""));
    try testing.expectEqualStrings("", try longestCommonSubstring(testing.allocator, "", "a"));
    try testing.expectEqualStrings("a", try longestCommonSubstring(testing.allocator, "a", "a"));
    try testing.expectEqualStrings("bcd", try longestCommonSubstring(testing.allocator, "abcdef", "bcd"));
    try testing.expectEqualStrings("ab", try longestCommonSubstring(testing.allocator, "abcdef", "xabded"));
    try testing.expectEqualStrings("Geeks", try longestCommonSubstring(testing.allocator, "GeeksforGeeks", "GeeksQuiz"));
    try testing.expectEqualStrings("abcd", try longestCommonSubstring(testing.allocator, "abcdxyz", "xyzabcd"));
    try testing.expectEqualStrings("abcdez", try longestCommonSubstring(testing.allocator, "zxabcdezy", "yzabcdezx"));
    try testing.expectEqualStrings(
        "Site:Geeks",
        try longestCommonSubstring(testing.allocator, "OldSite:GeeksforGeeks.org", "NewSite:GeeksQuiz.com"),
    );
}

test "longest common substring: repeated character extreme case" {
    var a: [4096]u8 = undefined;
    var b: [4096]u8 = undefined;
    @memset(&a, 'x');
    @memset(&b, 'x');

    const answer = try longestCommonSubstring(testing.allocator, &a, &b);
    try testing.expectEqual(@as(usize, 4096), answer.len);
}
