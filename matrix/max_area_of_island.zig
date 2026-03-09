//! Max Area Of Island - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/max_area_of_island.py

const std = @import("std");
const testing = std.testing;

pub const MaxAreaError = error{
    InvalidGrid,
    OutOfMemory,
};

const directions = [_][2]isize{
    .{ 1, 0 },
    .{ -1, 0 },
    .{ 0, 1 },
    .{ 0, -1 },
};

fn dfs(grid: []const []const u8, row: isize, col: isize, cols: usize, seen: []bool) usize {
    if (row < 0 or col < 0) return 0;

    const row_idx: usize = @intCast(row);
    const col_idx: usize = @intCast(col);
    if (row_idx >= grid.len or col_idx >= cols) return 0;

    const flat_idx = row_idx * cols + col_idx;
    if (seen[flat_idx] or grid[row_idx][col_idx] == 0) return 0;

    seen[flat_idx] = true;
    var area: usize = 1;
    for (directions) |delta| {
        area += dfs(grid, row + delta[0], col + delta[1], cols, seen);
    }
    return area;
}

/// Returns the largest 4-directionally connected island area.
///
/// Time complexity: O(r * c)
/// Space complexity: O(r * c)
pub fn findMaxArea(grid: []const []const u8, allocator: std.mem.Allocator) MaxAreaError!usize {
    if (grid.len == 0) return 0;
    const cols = grid[0].len;
    if (cols == 0) return 0;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    const seen = try allocator.alloc(bool, grid.len * cols);
    defer allocator.free(seen);
    @memset(seen, false);

    var max_area: usize = 0;
    for (grid, 0..) |row, row_idx| {
        for (row, 0..) |value, col_idx| {
            if (value == 1 and !seen[row_idx * cols + col_idx]) {
                max_area = @max(max_area, dfs(grid, @intCast(row_idx), @intCast(col_idx), cols, seen));
            }
        }
    }
    return max_area;
}

test "max area of island: python reference" {
    const allocator = testing.allocator;
    const matrix = [_][]const u8{
        &[_]u8{ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0 },
        &[_]u8{ 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
        &[_]u8{ 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0 },
    };

    try testing.expectEqual(@as(usize, 6), try findMaxArea(&matrix, allocator));
}

test "max area of island: boundaries" {
    const allocator = testing.allocator;

    const all_water = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 0 },
    };
    const diagonal = [_][]const u8{
        &[_]u8{ 1, 0 },
        &[_]u8{ 0, 1 },
    };
    const ragged = [_][]const u8{
        &[_]u8{ 1, 0 },
        &[_]u8{1},
    };

    try testing.expectEqual(@as(usize, 0), try findMaxArea(&all_water, allocator));
    try testing.expectEqual(@as(usize, 1), try findMaxArea(&diagonal, allocator));
    try testing.expectEqual(@as(usize, 0), try findMaxArea(&[_][]const u8{}, allocator));
    try testing.expectError(error.InvalidGrid, findMaxArea(&ragged, allocator));
}
