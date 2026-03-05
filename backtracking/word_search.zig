//! Word Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/word_search.py

const std = @import("std");
const testing = std.testing;

pub const WordSearchError = error{ InvalidBoard, InvalidWord };

fn pointKey(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

fn searchFrom(
    board: []const []const u8,
    word: []const u8,
    row: usize,
    col: usize,
    word_index: usize,
    visited: []bool,
    cols: usize,
) bool {
    if (board[row][col] != word[word_index]) return false;
    if (word_index == word.len - 1) return true;

    const key = pointKey(cols, row, col);
    visited[key] = true;
    defer visited[key] = false;

    const dirs = [_][2]isize{ .{ 0, 1 }, .{ 0, -1 }, .{ -1, 0 }, .{ 1, 0 } };
    for (dirs) |dir| {
        const next_row_i = @as(isize, @intCast(row)) + dir[0];
        const next_col_i = @as(isize, @intCast(col)) + dir[1];

        if (next_row_i < 0 or next_col_i < 0) continue;

        const next_row: usize = @intCast(next_row_i);
        const next_col: usize = @intCast(next_col_i);
        if (next_row >= board.len or next_col >= cols) continue;

        const next_key = pointKey(cols, next_row, next_col);
        if (visited[next_key]) continue;

        if (searchFrom(board, word, next_row, next_col, word_index + 1, visited, cols)) {
            return true;
        }
    }

    return false;
}

/// Returns true if `word` can be formed by adjacent (up/down/left/right) cells.
///
/// Time complexity: O(rows * cols * 4^L), where L is word length.
/// Space complexity: O(rows * cols) for visited flags + recursion stack.
pub fn wordExists(
    allocator: std.mem.Allocator,
    board: []const []const u8,
    word: []const u8,
) (WordSearchError || std.mem.Allocator.Error)!bool {
    if (board.len == 0) return WordSearchError.InvalidBoard;
    const cols = board[0].len;
    if (cols == 0) return WordSearchError.InvalidBoard;

    for (board) |row| {
        if (row.len == 0 or row.len != cols) return WordSearchError.InvalidBoard;
    }

    if (word.len == 0) return WordSearchError.InvalidWord;
    if (word.len > board.len * cols) return false;

    const visited = try allocator.alloc(bool, board.len * cols);
    defer allocator.free(visited);

    for (board, 0..) |row, r| {
        for (row, 0..) |_, c| {
            @memset(visited, false);
            if (searchFrom(board, word, r, c, 0, visited, cols)) {
                return true;
            }
        }
    }

    return false;
}

test "word search: standard examples" {
    const alloc = testing.allocator;
    const board = [_][]const u8{ "ABCE", "SFCS", "ADEE" };

    try testing.expect(try wordExists(alloc, &board, "ABCCED"));
    try testing.expect(try wordExists(alloc, &board, "SEE"));
    try testing.expect(!(try wordExists(alloc, &board, "ABCB")));
}

test "word search: single cell" {
    const alloc = testing.allocator;
    const board = [_][]const u8{"A"};

    try testing.expect(try wordExists(alloc, &board, "A"));
    try testing.expect(!(try wordExists(alloc, &board, "B")));
}

test "word search: cannot reuse same cell" {
    const alloc = testing.allocator;
    const board = [_][]const u8{ "BAA", "AAA", "ABA" };

    try testing.expect(!(try wordExists(alloc, &board, "ABB")));
}

test "word search: invalid board and word" {
    const alloc = testing.allocator;

    const empty_board = [_][]const u8{};
    try testing.expectError(WordSearchError.InvalidBoard, wordExists(alloc, &empty_board, "AB"));

    const bad_row_board = [_][]const u8{""};
    try testing.expectError(WordSearchError.InvalidBoard, wordExists(alloc, &bad_row_board, "AB"));

    const jagged = [_][]const u8{ "AB", "A" };
    try testing.expectError(WordSearchError.InvalidBoard, wordExists(alloc, &jagged, "AB"));

    const board = [_][]const u8{"AB"};
    try testing.expectError(WordSearchError.InvalidWord, wordExists(alloc, &board, ""));
}

test "word search: extreme length pruning" {
    const alloc = testing.allocator;
    const board = [_][]const u8{ "AB", "CD" };
    try testing.expect(!(try wordExists(alloc, &board, "ABCDE")));
}
