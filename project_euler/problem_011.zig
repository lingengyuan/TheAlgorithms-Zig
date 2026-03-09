//! Project Euler Problem 11: Largest Product in a Grid - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_011/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem011Error = error{
    GridTooSmall,
    NonRectangularGrid,
};

pub const default_grid = [20][20]u64{
    .{ 8, 2, 22, 97, 38, 15, 0, 40, 0, 75, 4, 5, 7, 78, 52, 12, 50, 77, 91, 8 },
    .{ 49, 49, 99, 40, 17, 81, 18, 57, 60, 87, 17, 40, 98, 43, 69, 48, 4, 56, 62, 0 },
    .{ 81, 49, 31, 73, 55, 79, 14, 29, 93, 71, 40, 67, 53, 88, 30, 3, 49, 13, 36, 65 },
    .{ 52, 70, 95, 23, 4, 60, 11, 42, 69, 24, 68, 56, 1, 32, 56, 71, 37, 2, 36, 91 },
    .{ 22, 31, 16, 71, 51, 67, 63, 89, 41, 92, 36, 54, 22, 40, 40, 28, 66, 33, 13, 80 },
    .{ 24, 47, 32, 60, 99, 3, 45, 2, 44, 75, 33, 53, 78, 36, 84, 20, 35, 17, 12, 50 },
    .{ 32, 98, 81, 28, 64, 23, 67, 10, 26, 38, 40, 67, 59, 54, 70, 66, 18, 38, 64, 70 },
    .{ 67, 26, 20, 68, 2, 62, 12, 20, 95, 63, 94, 39, 63, 8, 40, 91, 66, 49, 94, 21 },
    .{ 24, 55, 58, 5, 66, 73, 99, 26, 97, 17, 78, 78, 96, 83, 14, 88, 34, 89, 63, 72 },
    .{ 21, 36, 23, 9, 75, 0, 76, 44, 20, 45, 35, 14, 0, 61, 33, 97, 34, 31, 33, 95 },
    .{ 78, 17, 53, 28, 22, 75, 31, 67, 15, 94, 3, 80, 4, 62, 16, 14, 9, 53, 56, 92 },
    .{ 16, 39, 5, 42, 96, 35, 31, 47, 55, 58, 88, 24, 0, 17, 54, 24, 36, 29, 85, 57 },
    .{ 86, 56, 0, 48, 35, 71, 89, 7, 5, 44, 44, 37, 44, 60, 21, 58, 51, 54, 17, 58 },
    .{ 19, 80, 81, 68, 5, 94, 47, 69, 28, 73, 92, 13, 86, 52, 17, 77, 4, 89, 55, 40 },
    .{ 4, 52, 8, 83, 97, 35, 99, 16, 7, 97, 57, 32, 16, 26, 26, 79, 33, 27, 98, 66 },
    .{ 88, 36, 68, 87, 57, 62, 20, 72, 3, 46, 33, 67, 46, 55, 12, 32, 63, 93, 53, 69 },
    .{ 4, 42, 16, 73, 38, 25, 39, 11, 24, 94, 72, 18, 8, 46, 29, 32, 40, 62, 76, 36 },
    .{ 20, 69, 36, 41, 72, 30, 23, 88, 34, 62, 99, 69, 82, 67, 59, 85, 74, 4, 36, 16 },
    .{ 20, 73, 35, 29, 78, 31, 90, 1, 74, 31, 49, 71, 48, 86, 81, 16, 23, 57, 5, 54 },
    .{ 1, 70, 54, 71, 83, 51, 54, 69, 16, 92, 33, 48, 61, 43, 52, 1, 89, 19, 67, 48 },
};

/// Returns largest product of 4 adjacent entries in grid (horizontal,
/// vertical, and two diagonals).
///
/// Time complexity: O(r*c)
/// Space complexity: O(1)
pub fn largestProduct(grid: []const []const u64) Problem011Error!u64 {
    if (grid.len < 4 or grid[0].len < 4) return Problem011Error.GridTooSmall;

    const rows = grid.len;
    const cols = grid[0].len;
    for (grid) |row| {
        if (row.len != cols) return Problem011Error.NonRectangularGrid;
    }

    var largest: u64 = 0;

    for (0..rows) |r| {
        for (0..cols) |c| {
            if (c + 3 < cols) {
                const p = grid[r][c] * grid[r][c + 1] * grid[r][c + 2] * grid[r][c + 3];
                if (p > largest) largest = p;
            }
            if (r + 3 < rows) {
                const p = grid[r][c] * grid[r + 1][c] * grid[r + 2][c] * grid[r + 3][c];
                if (p > largest) largest = p;
            }
            if (r + 3 < rows and c + 3 < cols) {
                const p = grid[r][c] * grid[r + 1][c + 1] * grid[r + 2][c + 2] * grid[r + 3][c + 3];
                if (p > largest) largest = p;
            }
            if (r + 3 < rows and c >= 3) {
                const p = grid[r][c] * grid[r + 1][c - 1] * grid[r + 2][c - 2] * grid[r + 3][c - 3];
                if (p > largest) largest = p;
            }
        }
    }

    return largest;
}

/// Euler problem default solution.
pub fn solution() !u64 {
    var rows: [20][]const u64 = undefined;
    for (0..default_grid.len) |i| {
        rows[i] = default_grid[i][0..];
    }
    return largestProduct(&rows);
}

test "problem 011: python reference" {
    try testing.expectEqual(@as(u64, 70_600_674), try solution());
}

test "problem 011: boundaries and custom grid" {
    const tiny = [_][]const u64{
        &[_]u64{ 1, 2, 3 },
        &[_]u64{ 4, 5, 6 },
        &[_]u64{ 7, 8, 9 },
    };
    try testing.expectError(Problem011Error.GridTooSmall, largestProduct(&tiny));

    const non_rect = [_][]const u64{
        &[_]u64{ 1, 2, 3, 4 },
        &[_]u64{ 5, 6, 7 },
        &[_]u64{ 8, 9, 10, 11 },
        &[_]u64{ 12, 13, 14, 15 },
    };
    try testing.expectError(Problem011Error.NonRectangularGrid, largestProduct(&non_rect));

    const custom = [_][]const u64{
        &[_]u64{ 1, 1, 1, 2 },
        &[_]u64{ 1, 1, 3, 1 },
        &[_]u64{ 1, 4, 1, 1 },
        &[_]u64{ 5, 1, 1, 1 },
    };
    try testing.expectEqual(@as(u64, 120), try largestProduct(&custom));
}
