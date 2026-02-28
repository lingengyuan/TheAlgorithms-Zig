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

/// Solves the sudoku in-place. Returns true if solved, false if no solution.
pub fn solve(grid: *[9][9]u8) bool {
    for (0..9) |r| {
        for (0..9) |c| {
            if (grid[r][c] == 0) {
                for (1..10) |n| {
                    const digit: u8 = @intCast(n);
                    if (isSafe(grid.*, r, c, digit)) {
                        grid[r][c] = digit;
                        if (solve(grid)) return true;
                        grid[r][c] = 0;
                    }
                }
                return false;
            }
        }
    }
    return true; // No empty cell found → solved
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
