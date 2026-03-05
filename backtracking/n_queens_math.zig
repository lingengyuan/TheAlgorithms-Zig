//! N-Queens (Math/DFS Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/n_queens_math.py

const std = @import("std");
const testing = std.testing;

pub const Board = [][]u8;

pub fn freeBoards(allocator: std.mem.Allocator, boards: []Board) void {
    for (boards) |board| {
        for (board) |row| allocator.free(row);
        allocator.free(board);
    }
    allocator.free(boards);
}

fn buildBoard(allocator: std.mem.Allocator, positions: []const usize) std.mem.Allocator.Error!Board {
    const n = positions.len;
    const board = try allocator.alloc([]u8, n);
    errdefer allocator.free(board);

    var built: usize = 0;
    errdefer {
        for (board[0..built]) |row| allocator.free(row);
    }

    for (positions, 0..) |queen_col, row| {
        _ = row;
        const line = try allocator.alloc(u8, 2 * n);
        built += 1;
        for (0..n) |col| {
            line[2 * col] = if (col == queen_col) 'Q' else '.';
            line[2 * col + 1] = ' ';
        }
        board[built - 1] = line;
    }

    return board;
}

fn dfs(
    allocator: std.mem.Allocator,
    row: usize,
    n: usize,
    positions: []usize,
    used_cols: []bool,
    diag_right: []bool,
    diag_left: []bool,
    boards: *std.ArrayListUnmanaged(Board),
) std.mem.Allocator.Error!void {
    if (row == n) {
        try boards.append(allocator, try buildBoard(allocator, positions));
        return;
    }

    for (0..n) |col| {
        const right_idx = row + (n - 1 - col); // row - col + (n - 1)
        const left_idx = row + col; // row + col
        if (used_cols[col] or diag_right[right_idx] or diag_left[left_idx]) continue;

        used_cols[col] = true;
        diag_right[right_idx] = true;
        diag_left[left_idx] = true;
        positions[row] = col;

        try dfs(allocator, row + 1, n, positions, used_cols, diag_right, diag_left, boards);

        used_cols[col] = false;
        diag_right[right_idx] = false;
        diag_left[left_idx] = false;
    }
}

/// Returns all N-Queens boards in the same DFS order and row formatting as the Python reference.
/// Each row is formatted like `". Q . . "` with trailing spaces.
///
/// Time complexity: O(n!) worst-case search.
/// Space complexity: O(n) recursion + O(solution_size) output.
pub fn nQueensMathBoards(
    allocator: std.mem.Allocator,
    n: usize,
) std.mem.Allocator.Error![]Board {
    var boards = std.ArrayListUnmanaged(Board){};
    errdefer {
        for (boards.items) |board| {
            for (board) |row| allocator.free(row);
            allocator.free(board);
        }
        boards.deinit(allocator);
    }

    if (n == 0) {
        try boards.append(allocator, try allocator.alloc([]u8, 0));
        return boards.toOwnedSlice(allocator);
    }

    const positions = try allocator.alloc(usize, n);
    defer allocator.free(positions);

    const used_cols = try allocator.alloc(bool, n);
    defer allocator.free(used_cols);
    @memset(used_cols, false);

    const diag_right = try allocator.alloc(bool, 2 * n - 1);
    defer allocator.free(diag_right);
    @memset(diag_right, false);

    const diag_left = try allocator.alloc(bool, 2 * n - 1);
    defer allocator.free(diag_left);
    @memset(diag_left, false);

    try dfs(allocator, 0, n, positions, used_cols, diag_right, diag_left, &boards);
    return boards.toOwnedSlice(allocator);
}

test "n-queens math: python 4x4 doctest layout" {
    const alloc = testing.allocator;
    const boards = try nQueensMathBoards(alloc, 4);
    defer freeBoards(alloc, boards);

    try testing.expectEqual(@as(usize, 2), boards.len);
    const board1 = [_][]const u8{ ". Q . . ", ". . . Q ", "Q . . . ", ". . Q . " };
    const board2 = [_][]const u8{ ". . Q . ", "Q . . . ", ". . . Q ", ". Q . . " };

    for (board1, 0..) |row, i| try testing.expectEqualStrings(row, boards[0][i]);
    for (board2, 0..) |row, i| try testing.expectEqualStrings(row, boards[1][i]);
}

test "n-queens math: boundaries" {
    const alloc = testing.allocator;

    const one = try nQueensMathBoards(alloc, 1);
    defer freeBoards(alloc, one);
    try testing.expectEqual(@as(usize, 1), one.len);
    try testing.expectEqual(@as(usize, 1), one[0].len);
    try testing.expectEqualStrings("Q ", one[0][0]);

    const zero = try nQueensMathBoards(alloc, 0);
    defer freeBoards(alloc, zero);
    try testing.expectEqual(@as(usize, 1), zero.len);
    try testing.expectEqual(@as(usize, 0), zero[0].len);
}

test "n-queens math: extreme board count" {
    const alloc = testing.allocator;
    const boards = try nQueensMathBoards(alloc, 8);
    defer freeBoards(alloc, boards);
    try testing.expectEqual(@as(usize, 92), boards.len);
}
