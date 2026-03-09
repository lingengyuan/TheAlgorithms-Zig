//! Count Islands In Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/count_islands_in_matrix.py

const std = @import("std");
const testing = std.testing;

pub const CountIslandsError = error{
    InvalidGrid,
    OutOfMemory,
};

const directions = [_][2]isize{
    .{ -1, -1 }, .{ -1, 0 }, .{ -1, 1 },
    .{ 0, -1 },  .{ 0, 1 },  .{ 1, -1 },
    .{ 1, 0 },   .{ 1, 1 },
};

fn floodFill(grid: []const []const u8, row: isize, col: isize, cols: usize, visited: []bool) void {
    if (row < 0 or col < 0) return;

    const row_idx: usize = @intCast(row);
    const col_idx: usize = @intCast(col);
    if (row_idx >= grid.len or col_idx >= cols) return;

    const visit_idx = row_idx * cols + col_idx;
    if (visited[visit_idx] or grid[row_idx][col_idx] == 0) return;

    visited[visit_idx] = true;
    for (directions) |delta| {
        floodFill(grid, row + delta[0], col + delta[1], cols, visited);
    }
}

/// Counts 8-directionally connected islands in a binary matrix.
///
/// Time complexity: O(r * c)
/// Space complexity: O(r * c)
pub fn countIslands(grid: []const []const u8, allocator: std.mem.Allocator) CountIslandsError!usize {
    if (grid.len == 0) return 0;
    const cols = grid[0].len;
    if (cols == 0) return 0;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    const visited = try allocator.alloc(bool, grid.len * cols);
    defer allocator.free(visited);
    @memset(visited, false);

    var count: usize = 0;
    for (grid, 0..) |row, row_idx| {
        for (row, 0..) |value, col_idx| {
            const visit_idx = row_idx * cols + col_idx;
            if (value == 1 and !visited[visit_idx]) {
                floodFill(grid, @intCast(row_idx), @intCast(col_idx), cols, visited);
                count += 1;
            }
        }
    }
    return count;
}

test "count islands in matrix: reference example" {
    const allocator = testing.allocator;

    const grid = [_][]const u8{
        &[_]u8{ 1, 1, 0, 0, 0 },
        &[_]u8{ 0, 1, 0, 0, 1 },
        &[_]u8{ 1, 0, 0, 1, 1 },
        &[_]u8{ 0, 0, 0, 0, 0 },
        &[_]u8{ 1, 0, 1, 0, 1 },
    };

    try testing.expectEqual(@as(usize, 5), try countIslands(&grid, allocator));
}

test "count islands in matrix: boundaries" {
    const allocator = testing.allocator;

    const diagonal_chain = [_][]const u8{
        &[_]u8{ 1, 0, 0 },
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 0, 0, 1 },
    };
    const all_water = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 0 },
    };
    const ragged = [_][]const u8{
        &[_]u8{ 1, 0 },
        &[_]u8{1},
    };

    try testing.expectEqual(@as(usize, 1), try countIslands(&diagonal_chain, allocator));
    try testing.expectEqual(@as(usize, 0), try countIslands(&all_water, allocator));
    try testing.expectEqual(@as(usize, 0), try countIslands(&[_][]const u8{}, allocator));
    try testing.expectError(error.InvalidGrid, countIslands(&ragged, allocator));
}
