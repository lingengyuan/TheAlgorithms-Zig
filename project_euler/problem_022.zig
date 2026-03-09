//! Project Euler Problem 22: Names Scores - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_022/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem022Error = error{
    InvalidCharacter,
    OutOfMemory,
};

const default_names_csv = @embedFile("problem_022_names.txt");

fn lessName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Returns alphabetical value of a name (A=1, ..., Z=26).
pub fn nameValue(name: []const u8) Problem022Error!u64 {
    var total: u64 = 0;
    for (name) |ch| {
        if (ch < 'A' or ch > 'Z') return Problem022Error.InvalidCharacter;
        total += ch - 'A' + 1;
    }
    return total;
}

/// Parses CSV-style quoted names, sorts, and returns total name score.
///
/// Time complexity: O(m log m + total_chars)
/// Space complexity: O(m)
pub fn totalNameScores(csv: []const u8, allocator: std.mem.Allocator) Problem022Error!u64 {
    var names = std.ArrayListUnmanaged([]const u8){};
    defer names.deinit(allocator);

    var split = std.mem.splitScalar(u8, csv, ',');
    while (split.next()) |token| {
        const trimmed = std.mem.trim(u8, token, "\" \r\n\t");
        if (trimmed.len == 0) continue;

        _ = try nameValue(trimmed);
        try names.append(allocator, trimmed);
    }

    std.mem.sort([]const u8, names.items, {}, lessName);

    var total: u64 = 0;
    for (names.items, 0..) |name, idx| {
        const value = try nameValue(name);
        total += (idx + 1) * value;
    }

    return total;
}

/// Euler problem default solution.
pub fn solution(allocator: std.mem.Allocator) Problem022Error!u64 {
    return totalNameScores(default_names_csv, allocator);
}

test "problem 022: python reference" {
    try testing.expectEqual(@as(u64, 871_198_282), try solution(testing.allocator));
}

test "problem 022: sorting, boundaries, and validation" {
    const sample = "\"COLIN\",\"ALEX\",\"BOB\"";
    // Sorted: ALEX(42), BOB(19), COLIN(53) => 1*42 + 2*19 + 3*53 = 239
    try testing.expectEqual(@as(u64, 239), try totalNameScores(sample, testing.allocator));

    try testing.expectEqual(@as(u64, 0), try totalNameScores("", testing.allocator));
    try testing.expectEqual(@as(u64, 0), try totalNameScores("\n\r\t", testing.allocator));

    try testing.expectError(Problem022Error.InvalidCharacter, totalNameScores("\"ANNA1\"", testing.allocator));
    try testing.expectError(Problem022Error.InvalidCharacter, nameValue("alice"));
}
