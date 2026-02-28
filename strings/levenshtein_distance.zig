//! Levenshtein Distance - Zig implementation (space-optimised 1D DP)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/levenshtein_distance.py

const std = @import("std");
const testing = std.testing;

/// Computes the Levenshtein (edit) distance between two strings.
/// Uses a space-optimised 1-D DP array: O(min(m, n)) space.
/// Time complexity: O(m × n)
pub fn levenshteinDistance(allocator: std.mem.Allocator, a: []const u8, b: []const u8) !usize {
    // Ensure a is the shorter string for space optimisation
    if (a.len > b.len) return levenshteinDistance(allocator, b, a);

    const m = a.len;
    const n = b.len;

    // prev[j] = distance(a[0..i], b[0..j]) for the previous row
    const prev = try allocator.alloc(usize, n + 1);
    defer allocator.free(prev);
    const curr = try allocator.alloc(usize, n + 1);
    defer allocator.free(curr);

    for (0..n + 1) |j| prev[j] = j;

    for (0..m) |i| {
        curr[0] = i + 1;
        for (0..n) |j| {
            if (a[i] == b[j]) {
                curr[j + 1] = prev[j];
            } else {
                curr[j + 1] = 1 + @min(prev[j], @min(curr[j], prev[j + 1]));
            }
        }
        @memcpy(prev, curr);
    }
    return prev[n];
}

test "levenshtein: known pairs" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 3), try levenshteinDistance(alloc, "kitten", "sitting"));
    try testing.expectEqual(@as(usize, 5), try levenshteinDistance(alloc, "intention", "execution"));
    try testing.expectEqual(@as(usize, 0), try levenshteinDistance(alloc, "abc", "abc"));
}

test "levenshtein: empty strings" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try levenshteinDistance(alloc, "", ""));
    try testing.expectEqual(@as(usize, 5), try levenshteinDistance(alloc, "", "hello"));
    try testing.expectEqual(@as(usize, 3), try levenshteinDistance(alloc, "abc", ""));
}

test "levenshtein: single char" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try levenshteinDistance(alloc, "a", "b"));
    try testing.expectEqual(@as(usize, 0), try levenshteinDistance(alloc, "x", "x"));
}
