//! Count Paths - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/count_paths.py

const std = @import("std");
const testing = std.testing;

pub const CountPathsError = error{
    InvalidGrid,
    OutOfMemory,
};

fn dfs(grid: []const []const u8, row: isize, col: isize, cols: usize, visited: []bool) usize {
    if (row < 0 or col < 0) return 0;

    const row_idx: usize = @intCast(row);
    const col_idx: usize = @intCast(col);

    if (row_idx >= grid.len or col_idx >= cols) return 0;
    if (grid[row_idx][col_idx] == 1) return 0;

    const visit_idx = row_idx * cols + col_idx;
    if (visited[visit_idx]) return 0;

    if (row_idx == grid.len - 1 and col_idx == cols - 1) return 1;

    visited[visit_idx] = true;
    defer visited[visit_idx] = false;

    return dfs(grid, row + 1, col, cols, visited) +
        dfs(grid, row - 1, col, cols, visited) +
        dfs(grid, row, col + 1, cols, visited) +
        dfs(grid, row, col - 1, cols, visited);
}

/// Counts distinct paths from the top-left to the bottom-right of a blocked grid.
///
/// Time complexity: exponential in the number of open cells
/// Space complexity: O(r * c)
pub fn countPaths(grid: []const []const u8, allocator: std.mem.Allocator) CountPathsError!usize {
    if (grid.len == 0 or grid[0].len == 0) return error.InvalidGrid;
    const cols = grid[0].len;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    const visited = try allocator.alloc(bool, grid.len * cols);
    defer allocator.free(visited);
    @memset(visited, false);

    return dfs(grid, 0, 0, cols, visited);
}

test "count paths: python reference" {
    const allocator = testing.allocator;

    const grid1 = [_][]const u8{
        &[_]u8{ 0, 0, 0, 0 },
        &[_]u8{ 1, 1, 0, 0 },
        &[_]u8{ 0, 0, 0, 1 },
        &[_]u8{ 0, 1, 0, 0 },
    };
    const grid2 = [_][]const u8{
        &[_]u8{ 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 1, 1, 1, 0 },
        &[_]u8{ 0, 1, 1, 1, 0 },
        &[_]u8{ 0, 0, 0, 0, 0 },
    };

    try testing.expectEqual(@as(usize, 2), try countPaths(&grid1, allocator));
    try testing.expectEqual(@as(usize, 2), try countPaths(&grid2, allocator));
}

test "count paths: boundaries" {
    const allocator = testing.allocator;

    const blocked_start = [_][]const u8{
        &[_]u8{ 1, 0 },
        &[_]u8{ 0, 0 },
    };
    const blocked_finish = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 1 },
    };
    const single_open = [_][]const u8{&[_]u8{0}};
    const single_blocked = [_][]const u8{&[_]u8{1}};
    const ragged = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{0},
    };

    try testing.expectEqual(@as(usize, 0), try countPaths(&blocked_start, allocator));
    try testing.expectEqual(@as(usize, 0), try countPaths(&blocked_finish, allocator));
    try testing.expectEqual(@as(usize, 1), try countPaths(&single_open, allocator));
    try testing.expectEqual(@as(usize, 0), try countPaths(&single_blocked, allocator));
    try testing.expectError(error.InvalidGrid, countPaths(&[_][]const u8{}, allocator));
    try testing.expectError(error.InvalidGrid, countPaths(&ragged, allocator));
}
