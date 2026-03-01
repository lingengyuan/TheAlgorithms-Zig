//! Sudoku Solver - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/sudoku.py

const std = @import("std");
const testing = std.testing;

/// Returns true if placing `n` at (row, col) is valid in the 9×9 grid.
fn isSafe(grid: [9][9]u8, row: usize, col: usize, n: u8) bool {
    // Check row
    for (0..9) |c| {
        if (grid[row][c] == n) return false;
    }
    // Check column
    for (0..9) |r| {
        if (grid[r][col] == n) return false;
    }
    // Check 3×3 box
    const box_r = (row / 3) * 3;
    const box_c = (col / 3) * 3;
    for (box_r..box_r + 3) |r| {
        for (box_c..box_c + 3) |c| {
            if (grid[r][c] == n) return false;
        }
    }
    return true;
}

fn isExistingCellValid(grid: [9][9]u8, row: usize, col: usize) bool {
    const value = grid[row][col];
    if (value == 0) return true;
    if (value > 9) return false;

    for (0..9) |c| {
        if (c != col and grid[row][c] == value) return false;
    }
    for (0..9) |r| {
        if (r != row and grid[r][col] == value) return false;
    }

    const box_r = (row / 3) * 3;
    const box_c = (col / 3) * 3;
    for (box_r..box_r + 3) |r| {
        for (box_c..box_c + 3) |c| {
            if ((r != row or c != col) and grid[r][c] == value) return false;
        }
    }
    return true;
}

fn isGridValid(grid: [9][9]u8) bool {
    for (0..9) |r| {
        for (0..9) |c| {
            if (!isExistingCellValid(grid, r, c)) return false;
        }
    }
    return true;
}

fn solveRecursive(grid: *[9][9]u8) bool {
    for (0..9) |r| {
        for (0..9) |c| {
            if (grid[r][c] == 0) {
                for (1..10) |n| {
                    const digit: u8 = @intCast(n);
                    if (isSafe(grid.*, r, c, digit)) {
                        grid[r][c] = digit;
                        if (solveRecursive(grid)) return true;
                        grid[r][c] = 0;
                    }
                }
                return false;
            }
        }
    }
    return true; // No empty cell found → solved
}

/// Solves the sudoku in-place. Returns true if solved, false if no solution.
pub fn solve(grid: *[9][9]u8) bool {
    if (!isGridValid(grid.*)) return false;
    return solveRecursive(grid);
}

test "sudoku: solvable puzzle" {
    var grid = [9][9]u8{
        [_]u8{ 3, 0, 6, 5, 0, 8, 4, 0, 0 },
        [_]u8{ 5, 2, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 8, 7, 0, 0, 0, 0, 3, 1 },
        [_]u8{ 0, 0, 3, 0, 1, 0, 0, 8, 0 },
        [_]u8{ 9, 0, 0, 8, 6, 3, 0, 0, 5 },
        [_]u8{ 0, 5, 0, 0, 9, 0, 6, 0, 0 },
        [_]u8{ 1, 3, 0, 0, 0, 0, 2, 5, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 7, 4 },
        [_]u8{ 0, 0, 5, 2, 0, 6, 3, 0, 0 },
    };
    try testing.expect(solve(&grid));
    // No zeros remain
    for (grid) |row| {
        for (row) |cell| try testing.expect(cell != 0);
    }
    // Spot-check: row 0 must have no zeros and sum to 45 (1-9 each once)
    var row0_sum: u32 = 0;
    for (grid[0]) |cell| row0_sum += cell;
    try testing.expectEqual(@as(u32, 45), row0_sum);
}

test "sudoku: unsolvable puzzle" {
    // Two 5s in the same row — immediately unsolvable
    var grid = [9][9]u8{
        [_]u8{ 5, 0, 6, 5, 0, 8, 4, 0, 3 },
        [_]u8{ 5, 2, 0, 0, 0, 0, 0, 0, 2 },
        [_]u8{ 1, 8, 7, 0, 0, 0, 0, 3, 1 },
        [_]u8{ 0, 0, 3, 0, 1, 0, 0, 8, 0 },
        [_]u8{ 9, 0, 0, 8, 6, 3, 0, 0, 5 },
        [_]u8{ 0, 5, 0, 0, 9, 0, 6, 0, 0 },
        [_]u8{ 1, 3, 0, 0, 0, 0, 2, 5, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 7, 4 },
        [_]u8{ 0, 0, 5, 2, 0, 6, 3, 0, 0 },
    };
    try testing.expect(!solve(&grid));
}

test "sudoku: invalid but full grid is rejected" {
    var grid = [9][9]u8{
        [_]u8{ 1, 1, 2, 3, 4, 5, 6, 7, 8 },
        [_]u8{ 3, 4, 5, 6, 7, 8, 9, 1, 2 },
        [_]u8{ 6, 7, 8, 9, 1, 2, 3, 4, 5 },
        [_]u8{ 2, 3, 4, 5, 6, 7, 8, 9, 1 },
        [_]u8{ 5, 6, 7, 8, 9, 1, 2, 3, 4 },
        [_]u8{ 8, 9, 1, 2, 3, 4, 5, 6, 7 },
        [_]u8{ 4, 5, 6, 7, 8, 9, 1, 2, 3 },
        [_]u8{ 7, 8, 9, 1, 2, 3, 4, 5, 6 },
        [_]u8{ 9, 2, 3, 4, 5, 6, 7, 8, 9 },
    };
    try testing.expect(!solve(&grid));
}

test "sudoku: out of range digit is rejected" {
    var grid = [9][9]u8{
        [_]u8{ 3, 0, 6, 5, 0, 8, 4, 0, 0 },
        [_]u8{ 5, 2, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 8, 7, 0, 0, 0, 0, 3, 1 },
        [_]u8{ 0, 0, 3, 0, 1, 0, 0, 8, 0 },
        [_]u8{ 9, 0, 0, 8, 6, 3, 0, 0, 5 },
        [_]u8{ 0, 5, 0, 0, 9, 0, 6, 0, 0 },
        [_]u8{ 1, 3, 0, 0, 0, 0, 2, 5, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 7, 4 },
        [_]u8{ 0, 0, 5, 2, 0, 6, 3, 0, 15 },
    };
    try testing.expect(!solve(&grid));
}
