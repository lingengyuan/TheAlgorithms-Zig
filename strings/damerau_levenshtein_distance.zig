//! Damerau-Levenshtein Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/damerau_levenshtein_distance.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the Damerau-Levenshtein edit distance between two strings.
/// Time complexity: O(m × n), Space complexity: O(m × n)
pub fn damerauLevenshteinDistance(
    allocator: Allocator,
    first: []const u8,
    second: []const u8,
) !usize {
    const rows = first.len + 1;
    const cols = second.len + 1;
    const matrix = try allocator.alloc(usize, rows * cols);
    defer allocator.free(matrix);

    const Context = struct {
        fn at(buffer: []usize, cols_inner: usize, row: usize, col: usize) *usize {
            return &buffer[row * cols_inner + col];
        }
    };

    for (0..rows) |row| Context.at(matrix, cols, row, 0).* = row;
    for (0..cols) |col| Context.at(matrix, cols, 0, col).* = col;

    for (1..rows) |row| {
        for (1..cols) |col| {
            const cost: usize = if (first[row - 1] == second[col - 1]) 0 else 1;
            var best = @min(
                Context.at(matrix, cols, row - 1, col).* + 1,
                @min(
                    Context.at(matrix, cols, row, col - 1).* + 1,
                    Context.at(matrix, cols, row - 1, col - 1).* + cost,
                ),
            );
            if (row > 1 and col > 1 and first[row - 1] == second[col - 2] and first[row - 2] == second[col - 1]) {
                best = @min(best, Context.at(matrix, cols, row - 2, col - 2).* + cost);
            }
            Context.at(matrix, cols, row, col).* = best;
        }
    }

    return Context.at(matrix, cols, rows - 1, cols - 1).*;
}

test "damerau levenshtein distance: python samples" {
    try testing.expectEqual(@as(usize, 1), try damerauLevenshteinDistance(testing.allocator, "cat", "cut"));
    try testing.expectEqual(@as(usize, 3), try damerauLevenshteinDistance(testing.allocator, "kitten", "sitting"));
    try testing.expectEqual(@as(usize, 4), try damerauLevenshteinDistance(testing.allocator, "hello", "world"));
    try testing.expectEqual(@as(usize, 2), try damerauLevenshteinDistance(testing.allocator, "book", "back"));
    try testing.expectEqual(@as(usize, 3), try damerauLevenshteinDistance(testing.allocator, "container", "containment"));
}

test "damerau levenshtein distance: edge cases" {
    try testing.expectEqual(@as(usize, 0), try damerauLevenshteinDistance(testing.allocator, "", ""));
    try testing.expectEqual(@as(usize, 1), try damerauLevenshteinDistance(testing.allocator, "ab", "ba"));
    try testing.expectEqual(@as(usize, 4), try damerauLevenshteinDistance(testing.allocator, "test", ""));
}
