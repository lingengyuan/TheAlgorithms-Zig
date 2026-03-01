//! Levenshtein Distance - Zig implementation (space-optimised 1D DP)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/levenshtein_distance.py

const std = @import("std");
const testing = std.testing;

pub const LevenshteinError = error{Overflow};

/// Computes the Levenshtein (edit) distance between two strings.
/// Uses a space-optimised 1-D DP array: O(min(m, n)) space.
/// Time complexity: O(m × n)
pub fn levenshteinDistance(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) (LevenshteinError || std.mem.Allocator.Error)!usize {
    // Ensure a is the shorter string for space optimisation
    if (a.len > b.len) return levenshteinDistance(allocator, b, a);

    const m = a.len;
    const n = b.len;
    const n_plus = @addWithOverflow(n, @as(usize, 1));
    if (n_plus[1] != 0) return LevenshteinError.Overflow;

    // prev[j] = distance(a[0..i], b[0..j]) for the previous row
    const prev = try allocator.alloc(usize, n_plus[0]);
    defer allocator.free(prev);
    const curr = try allocator.alloc(usize, n_plus[0]);
    defer allocator.free(curr);

    for (0..n_plus[0]) |j| prev[j] = j;

    for (0..m) |i| {
        const next_i = @addWithOverflow(i, @as(usize, 1));
        if (next_i[1] != 0) return LevenshteinError.Overflow;
        curr[0] = next_i[0];
        for (0..n) |j| {
            if (a[i] == b[j]) {
                curr[j + 1] = prev[j];
            } else {
                const next = @addWithOverflow(@as(usize, 1), @min(prev[j], @min(curr[j], prev[j + 1])));
                if (next[1] != 0) return LevenshteinError.Overflow;
                curr[j + 1] = next[0];
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

test "levenshtein: oversize dimensions return overflow" {
    const fake_ptr: [*]const u8 = @ptrFromInt(@alignOf(u8));
    const fake = fake_ptr[0..std.math.maxInt(usize)];
    try testing.expectError(LevenshteinError.Overflow, levenshteinDistance(testing.allocator, "", fake));
}
