//! Edit Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/edit_distance.py

const std = @import("std");
const testing = std.testing;
const base = @import("levenshtein_distance.zig");

pub const EditDistanceError = base.LevenshteinError || std.mem.Allocator.Error;

/// Compatibility wrapper for the recursive Python file name.
/// Returns the minimum number of insert/delete/substitute operations.
/// Time complexity: O(m × n), Space complexity: O(min(m, n))
pub fn editDistance(
    allocator: std.mem.Allocator,
    source: []const u8,
    target: []const u8,
) EditDistanceError!usize {
    return base.levenshteinDistance(allocator, source, target);
}

test "edit distance: python samples" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try editDistance(allocator, "GATTIC", "GALTIC"));
    try testing.expectEqual(@as(usize, 2), try editDistance(allocator, "NUM3", "HUM2"));
    try testing.expectEqual(@as(usize, 3), try editDistance(allocator, "cap", "CAP"));
    try testing.expectEqual(@as(usize, 3), try editDistance(allocator, "Cat", ""));
}

test "edit distance: edge and extreme" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try editDistance(allocator, "cat", "cat"));
    try testing.expectEqual(@as(usize, 9), try editDistance(allocator, "", "123456789"));
    try testing.expectEqual(@as(usize, 5), try editDistance(allocator, "Be@uty", "Beautyyyy!"));
    try testing.expectEqual(@as(usize, 1), try editDistance(allocator, "lstring", "lsstring"));
}
