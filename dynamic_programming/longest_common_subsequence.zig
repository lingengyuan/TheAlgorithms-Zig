//! Longest Common Subsequence (LCS) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_common_subsequence.py

const std = @import("std");
const testing = std.testing;

pub const LcsResult = struct {
    length: usize,
    sequence: []u8,
};

/// Returns the longest common subsequence of two byte slices.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn longestCommonSubsequence(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) !LcsResult {
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

    const length = table[a.len * cols + b.len];
    const sequence = try allocator.alloc(u8, length);

    var back_i = a.len;
    var back_j = b.len;
    var idx = length;

    while (back_i > 0 and back_j > 0) {
        if (a[back_i - 1] == b[back_j - 1]) {
            idx -= 1;
            sequence[idx] = a[back_i - 1];
            back_i -= 1;
            back_j -= 1;
        } else {
            const up = table[(back_i - 1) * cols + back_j];
            const left = table[back_i * cols + (back_j - 1)];
            if (up >= left) {
                back_i -= 1;
            } else {
                back_j -= 1;
            }
        }
    }

    return .{
        .length = length,
        .sequence = sequence,
    };
}

pub fn longestCommonSubsequenceLength(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) !usize {
    const result = try longestCommonSubsequence(allocator, a, b);
    defer allocator.free(result.sequence);
    return result.length;
}

test "lcs: classic example" {
    const alloc = testing.allocator;
    const result = try longestCommonSubsequence(alloc, "AGGTAB", "GXTXAYB");
    defer alloc.free(result.sequence);
    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqualStrings("GTAB", result.sequence);
}

test "lcs: empty string" {
    const alloc = testing.allocator;
    const result = try longestCommonSubsequence(alloc, "", "ABC");
    defer alloc.free(result.sequence);
    try testing.expectEqual(@as(usize, 0), result.length);
    try testing.expectEqualStrings("", result.sequence);
}

test "lcs: no common subsequence" {
    const alloc = testing.allocator;
    const result = try longestCommonSubsequence(alloc, "ABC", "DEF");
    defer alloc.free(result.sequence);
    try testing.expectEqual(@as(usize, 0), result.length);
    try testing.expectEqualStrings("", result.sequence);
}

test "lcs: identical strings" {
    const alloc = testing.allocator;
    const result = try longestCommonSubsequence(alloc, "HELLO", "HELLO");
    defer alloc.free(result.sequence);
    try testing.expectEqual(@as(usize, 5), result.length);
    try testing.expectEqualStrings("HELLO", result.sequence);
}

test "lcs: subsequence case" {
    const alloc = testing.allocator;
    const result = try longestCommonSubsequence(alloc, "AXBYCZ", "ABC");
    defer alloc.free(result.sequence);
    try testing.expectEqual(@as(usize, 3), result.length);
    try testing.expectEqualStrings("ABC", result.sequence);
}

test "lcs: python doc example" {
    const alloc = testing.allocator;
    const result = try longestCommonSubsequence(alloc, "programming", "gaming");
    defer alloc.free(result.sequence);
    try testing.expectEqual(@as(usize, 6), result.length);
    try testing.expectEqualStrings("gaming", result.sequence);
}
