//! Conway's Game of Life - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/cellular_automata/conways_game_of_life.py

const std = @import("std");
const testing = std.testing;

pub const GameOfLifeError = error{
    EmptyGrid,
    NonRectangularGrid,
};

fn validateGrid(cells: []const []const u8) GameOfLifeError!usize {
    if (cells.len == 0 or cells[0].len == 0) {
        return GameOfLifeError.EmptyGrid;
    }
    const width = cells[0].len;
    for (cells) |row| {
        if (row.len != width) {
            return GameOfLifeError.NonRectangularGrid;
        }
    }
    return width;
}

/// Computes next generation for Conway's Game of Life.
/// Caller owns returned rows and each row slice.
///
/// Time complexity: O(h*w)
/// Space complexity: O(h*w)
pub fn newGeneration(allocator: std.mem.Allocator, cells: []const []const u8) ![][]u8 {
    const width = try validateGrid(cells);
    const height = cells.len;

    const next = try allocator.alloc([]u8, height);
    errdefer allocator.free(next);

    var built_rows: usize = 0;
    errdefer {
        for (0..built_rows) |i| allocator.free(next[i]);
    }

    for (cells, 0..) |_, i| {
        next[i] = try allocator.alloc(u8, width);
        built_rows += 1;

        for (0..width) |j| {
            var neighbours: u8 = 0;
            const i_start = if (i == 0) 0 else i - 1;
            const i_end = @min(i + 1, height - 1);
            const j_start = if (j == 0) 0 else j - 1;
            const j_end = @min(j + 1, width - 1);

            var y = i_start;
            while (y <= i_end) : (y += 1) {
                var x = j_start;
                while (x <= j_end) : (x += 1) {
                    if (y == i and x == j) continue;
                    neighbours += cells[y][x];
                }
            }

            const alive = cells[i][j] == 1;
            if ((alive and neighbours >= 2 and neighbours <= 3) or (!alive and neighbours == 3)) {
                next[i][j] = 1;
            } else {
                next[i][j] = 0;
            }
        }
    }

    return next;
}

fn freeGrid(allocator: std.mem.Allocator, grid: [][]u8) void {
    for (grid) |row| allocator.free(row);
    allocator.free(grid);
}

test "conway game of life: python blinker example" {
    const alloc = testing.allocator;

    const blinker = [_][]const u8{
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 0, 1, 0 },
    };

    const next = try newGeneration(alloc, &blinker);
    defer freeGrid(alloc, next);

    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, next[0]);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1 }, next[1]);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, next[2]);
}

test "conway game of life: validation and extreme values" {
    const alloc = testing.allocator;

    try testing.expectError(GameOfLifeError.EmptyGrid, newGeneration(alloc, &[_][]const u8{}));

    const non_rect = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{1},
    };
    try testing.expectError(GameOfLifeError.NonRectangularGrid, newGeneration(alloc, &non_rect));

    var large_rows: [64][]const u8 = undefined;
    const zero_row = [_]u8{0} ** 64;
    for (0..64) |i| {
        large_rows[i] = &zero_row;
    }
    const large_next = try newGeneration(alloc, &large_rows);
    defer freeGrid(alloc, large_next);
    for (large_next) |row| {
        for (row) |cell| {
            try testing.expectEqual(@as(u8, 0), cell);
        }
    }
}
