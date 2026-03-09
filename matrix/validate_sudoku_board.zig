//! Validate Sudoku Board - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/validate_sudoku_board.py

const std = @import("std");
const testing = std.testing;

pub const ValidateSudokuError = error{
    InvalidDimensions,
};

const num_squares = 9;
const empty_cell: u8 = '.';

/// Validates a partially filled 9x9 Sudoku board without solving it.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn isValidSudokuBoard(board: []const []const u8) ValidateSudokuError!bool {
    if (board.len != num_squares) return error.InvalidDimensions;
    for (board) |row| {
        if (row.len != num_squares) return error.InvalidDimensions;
    }

    var row_seen = std.mem.zeroes([num_squares][256]bool);
    var col_seen = std.mem.zeroes([num_squares][256]bool);
    var box_seen = std.mem.zeroes([num_squares][256]bool);

    for (board, 0..) |row, row_idx| {
        for (row, 0..) |value, col_idx| {
            if (value == empty_cell) continue;

            const box_idx = (row_idx / 3) * 3 + (col_idx / 3);
            const key: usize = value;
            if (row_seen[row_idx][key] or col_seen[col_idx][key] or box_seen[box_idx][key]) {
                return false;
            }

            row_seen[row_idx][key] = true;
            col_seen[col_idx][key] = true;
            box_seen[box_idx][key] = true;
        }
    }
    return true;
}

test "validate sudoku board: python reference" {
    const valid_board = [_][]const u8{
        "53..7....",
        "6..195...",
        ".98....6.",
        "8...6...3",
        "4..8.3..1",
        "7...2...6",
        ".6....28.",
        "...419..5",
        "....8..79",
    };
    const invalid_board = [_][]const u8{
        "83..7....",
        "6..195...",
        ".98....6.",
        "8...6...3",
        "4..8.3..1",
        "7...2...6",
        ".6....28.",
        "...419..5",
        "....8..79",
    };

    try testing.expect(try isValidSudokuBoard(&valid_board));
    try testing.expect(!(try isValidSudokuBoard(&invalid_board)));
}

test "validate sudoku board: boundaries" {
    const valid_sparse = [_][]const u8{
        "123......",
        "456......",
        "789......",
        "...456...",
        "...789...",
        "...123...",
        "......789",
        "......123",
        "......456",
    };
    const invalid_column = [_][]const u8{
        "123456789",
        "2.......8",
        "3.......7",
        "4.......6",
        "5.......5",
        "6.......4",
        "7.......3",
        "8.......2",
        "987654321",
    };

    try testing.expect(try isValidSudokuBoard(&valid_sparse));
    try testing.expect(!(try isValidSudokuBoard(&invalid_column)));
    try testing.expectError(error.InvalidDimensions, isValidSudokuBoard(&[_][]const u8{"123456789"}));
}
