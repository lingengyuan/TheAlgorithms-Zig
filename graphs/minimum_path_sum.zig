//! Minimum Path Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/minimum_path_sum.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes the minimum path sum from top-left to bottom-right in a grid.
/// Allowed moves are only right and down.
/// Returns `error.InvalidGrid` for empty/ragged grids.
/// Time complexity: O(rows * cols), Space complexity: O(cols)
pub fn minPathSum(allocator: Allocator, grid: []const []const i64) !i64 {
    if (grid.len == 0) return error.InvalidGrid;
    const cols = grid[0].len;
    if (cols == 0) return error.InvalidGrid;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    const dp = try allocator.alloc(i64, cols);
    defer allocator.free(dp);

    dp[0] = grid[0][0];
    for (1..cols) |c| {
        const sum = @addWithOverflow(dp[c - 1], grid[0][c]);
        if (sum[1] != 0) return error.Overflow;
        dp[c] = sum[0];
    }

    for (1..grid.len) |r| {
        const first = @addWithOverflow(dp[0], grid[r][0]);
        if (first[1] != 0) return error.Overflow;
        dp[0] = first[0];

        for (1..cols) |c| {
            const best_prev = if (dp[c - 1] < dp[c]) dp[c - 1] else dp[c];
            const sum = @addWithOverflow(best_prev, grid[r][c]);
            if (sum[1] != 0) return error.Overflow;
            dp[c] = sum[0];
        }
    }

    return dp[cols - 1];
}

test "minimum path sum: python examples" {
    const alloc = testing.allocator;
    const g1 = [_][]const i64{
        &[_]i64{ 1, 3, 1 },
        &[_]i64{ 1, 5, 1 },
        &[_]i64{ 4, 2, 1 },
    };
    try testing.expectEqual(@as(i64, 7), try minPathSum(alloc, &g1));

    const g2 = [_][]const i64{
        &[_]i64{ 1, 0, 5, 6, 7 },
        &[_]i64{ 8, 9, 0, 4, 2 },
        &[_]i64{ 4, 4, 4, 5, 1 },
        &[_]i64{ 9, 6, 3, 1, 0 },
        &[_]i64{ 8, 4, 3, 2, 7 },
    };
    try testing.expectEqual(@as(i64, 20), try minPathSum(alloc, &g2));
}

test "minimum path sum: invalid grid cases" {
    const alloc = testing.allocator;
    const empty = [_][]const i64{};
    try testing.expectError(error.InvalidGrid, minPathSum(alloc, &empty));

    const zero_col = [_][]const i64{
        &[_]i64{},
    };
    try testing.expectError(error.InvalidGrid, minPathSum(alloc, &zero_col));

    const ragged = [_][]const i64{
        &[_]i64{ 1, 2 },
        &[_]i64{1},
    };
    try testing.expectError(error.InvalidGrid, minPathSum(alloc, &ragged));
}

test "minimum path sum: supports negative values" {
    const alloc = testing.allocator;
    const g = [_][]const i64{
        &[_]i64{ 1, -3, 2 },
        &[_]i64{ 2, -1, 1 },
        &[_]i64{ 3, 2, -5 },
    };
    try testing.expectEqual(@as(i64, -7), try minPathSum(alloc, &g));
}

test "minimum path sum: extreme large grid" {
    const alloc = testing.allocator;
    const rows: usize = 120;
    const cols: usize = 120;

    const mut = try alloc.alloc([]i64, rows);
    defer {
        for (mut) |row| alloc.free(row);
        alloc.free(mut);
    }
    for (0..rows) |r| {
        mut[r] = try alloc.alloc(i64, cols);
        for (0..cols) |c| mut[r][c] = 1;
    }

    const grid = try alloc.alloc([]const i64, rows);
    defer alloc.free(grid);
    for (mut, 0..) |row, i| grid[i] = row;

    try testing.expectEqual(@as(i64, @intCast(rows + cols - 1)), try minPathSum(alloc, grid));
}
