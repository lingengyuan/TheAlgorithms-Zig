//! Edit Distance (Levenshtein Distance) - Zig implementation (bottom-up DP)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/edit_distance.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes the minimum edit distance between two strings.
/// Operations: insert, delete, substitute (each costs 1).
/// Time complexity: O(m × n), Space complexity: O(m × n)
pub fn editDistance(allocator: Allocator, word1: []const u8, word2: []const u8) !usize {
    const m = word1.len;
    const n = word2.len;

    // Allocate flattened 2D table of size (m+1) × (n+1)
    const dp = try allocator.alloc(usize, (m + 1) * (n + 1));
    defer allocator.free(dp);

    const cols = n + 1;

    // Base cases
    for (0..m + 1) |i| dp[i * cols] = i;
    for (0..n + 1) |j| dp[j] = j;

    // Fill table
    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i * cols + j] = dp[(i - 1) * cols + (j - 1)];
            } else {
                const ins = dp[i * cols + (j - 1)];
                const del = dp[(i - 1) * cols + j];
                const rep = dp[(i - 1) * cols + (j - 1)];
                dp[i * cols + j] = 1 + @min(ins, @min(del, rep));
            }
        }
    }

    return dp[m * cols + n];
}

// ===== Tests =====

test "edit distance: intention -> execution" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 5), try editDistance(alloc, "intention", "execution"));
}

test "edit distance: empty strings" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try editDistance(alloc, "", ""));
}

test "edit distance: one empty" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 9), try editDistance(alloc, "intention", ""));
    try testing.expectEqual(@as(usize, 5), try editDistance(alloc, "", "hello"));
}

test "edit distance: same strings" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try editDistance(alloc, "abc", "abc"));
}

test "edit distance: single char" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try editDistance(alloc, "a", "b"));
    try testing.expectEqual(@as(usize, 0), try editDistance(alloc, "a", "a"));
}

test "edit distance: kitten -> sitting" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 3), try editDistance(alloc, "kitten", "sitting"));
}
