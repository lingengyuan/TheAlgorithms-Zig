//! Longest Common Subsequence (LCS) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_common_subsequence.py

const std = @import("std");
const testing = std.testing;

/// Returns the length of the longest common subsequence of two byte slices.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn longestCommonSubsequenceLength(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) !usize {
    const rows = a.len + 1;
    const cols = b.len + 1;
    const table = try allocator.alloc(usize, rows * cols);
    defer allocator.free(table);
    @memset(table, 0);

    var i: usize = 1;
    while (i <= a.len) : (i += 1) {
        var j: usize = 1;
        while (j <= b.len) : (j += 1) {
            const idx = i * cols + j;
            const up = (i - 1) * cols + j;
            const left = i * cols + (j - 1);
            const diag = (i - 1) * cols + (j - 1);

            if (a[i - 1] == b[j - 1]) {
                table[idx] = table[diag] + 1;
            } else {
                table[idx] = @max(table[up], table[left]);
            }
        }
    }

    return table[a.len * cols + b.len];
}

test "lcs: classic example" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 4),
        try longestCommonSubsequenceLength(alloc, "AGGTAB", "GXTXAYB"),
    );
}

test "lcs: empty string" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 0),
        try longestCommonSubsequenceLength(alloc, "", "ABC"),
    );
}

test "lcs: no common subsequence" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 0),
        try longestCommonSubsequenceLength(alloc, "ABC", "DEF"),
    );
}

test "lcs: identical strings" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 5),
        try longestCommonSubsequenceLength(alloc, "HELLO", "HELLO"),
    );
}

test "lcs: subsequence case" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 3),
        try longestCommonSubsequenceLength(alloc, "AXBYCZ", "ABC"),
    );
}
