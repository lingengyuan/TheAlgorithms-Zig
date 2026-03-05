//! Rat in a Maze - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/rat_in_maze.py

const std = @import("std");
const testing = std.testing;

pub const MazeError = error{ InvalidMaze, InvalidCoordinates, NoSolution };

fn indexOf(n: usize, row: usize, col: usize) usize {
    return row * n + col;
}

fn runMaze(
    maze: []const []const u8,
    row: isize,
    col: isize,
    destination_row: usize,
    destination_col: usize,
    solution: []u8,
    n: usize,
) bool {
    if (row < 0 or col < 0) return false;

    const r: usize = @intCast(row);
    const c: usize = @intCast(col);
    if (r >= n or c >= n) return false;

    if (r == destination_row and c == destination_col and maze[r][c] == 0) {
        solution[indexOf(n, r, c)] = 0;
        return true;
    }

    const pos = indexOf(n, r, c);
    if (solution[pos] == 0 or maze[r][c] != 0) return false;

    solution[pos] = 0;
    if (runMaze(maze, row + 1, col, destination_row, destination_col, solution, n) or
        runMaze(maze, row, col + 1, destination_row, destination_col, solution, n) or
        runMaze(maze, row - 1, col, destination_row, destination_col, solution, n) or
        runMaze(maze, row, col - 1, destination_row, destination_col, solution, n))
    {
        return true;
    }

    solution[pos] = 1;
    return false;
}

/// Solves a binary maze where `0` is open path and `1` is blocked.
/// Returns a flattened `n*n` solution grid with path cells marked `0`.
///
/// Time complexity: O(n^2) average, O(4^(n^2)) worst case (backtracking).
/// Space complexity: O(n^2) for solution + recursion stack.
pub fn solveMaze(
    allocator: std.mem.Allocator,
    maze: []const []const u8,
    source_row: usize,
    source_col: usize,
    destination_row: usize,
    destination_col: usize,
) (MazeError || std.mem.Allocator.Error)![]u8 {
    const n = maze.len;
    if (n == 0) return MazeError.InvalidMaze;

    for (maze) |row| {
        if (row.len != n) return MazeError.InvalidMaze;
        for (row) |cell| {
            if (cell != 0 and cell != 1) return MazeError.InvalidMaze;
        }
    }

    if (source_row >= n or source_col >= n or destination_row >= n or destination_col >= n) {
        return MazeError.InvalidCoordinates;
    }

    const solution = try allocator.alloc(u8, n * n);
    errdefer allocator.free(solution);
    @memset(solution, 1);

    const solved = runMaze(
        maze,
        @intCast(source_row),
        @intCast(source_col),
        destination_row,
        destination_col,
        solution,
        n,
    );

    if (!solved) return MazeError.NoSolution;
    return solution;
}

test "rat in maze: standard solvable case" {
    const alloc = testing.allocator;
    const maze = [_][]const u8{
        &[_]u8{ 0, 0, 0 },
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 1, 0, 0 },
    };

    const solution = try solveMaze(alloc, &maze, 0, 0, 2, 2);
    defer alloc.free(solution);

    const expected = [_]u8{
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
    };
    try testing.expectEqualSlices(u8, &expected, solution);
}

test "rat in maze: no solution" {
    const alloc = testing.allocator;
    const maze = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 1, 1 },
    };

    try testing.expectError(MazeError.NoSolution, solveMaze(alloc, &maze, 0, 0, 1, 1));
}

test "rat in maze: invalid coordinates" {
    const alloc = testing.allocator;
    const maze = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{ 1, 0 },
    };

    try testing.expectError(MazeError.InvalidCoordinates, solveMaze(alloc, &maze, 2, 0, 1, 1));
}

test "rat in maze: invalid maze shape and values" {
    const alloc = testing.allocator;

    const empty = [_][]const u8{};
    try testing.expectError(MazeError.InvalidMaze, solveMaze(alloc, &empty, 0, 0, 0, 0));

    const jagged = [_][]const u8{
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 0, 0 },
        &[_]u8{ 1, 0, 0 },
    };
    try testing.expectError(MazeError.InvalidMaze, solveMaze(alloc, &jagged, 0, 0, 2, 2));

    const invalid_value = [_][]const u8{
        &[_]u8{ 0, 2 },
        &[_]u8{ 1, 0 },
    };
    try testing.expectError(MazeError.InvalidMaze, solveMaze(alloc, &invalid_value, 0, 0, 1, 1));
}

test "rat in maze: extreme one-cell cases" {
    const alloc = testing.allocator;

    const open = [_][]const u8{&[_]u8{0}};
    const solution = try solveMaze(alloc, &open, 0, 0, 0, 0);
    defer alloc.free(solution);
    try testing.expectEqualSlices(u8, &[_]u8{0}, solution);

    const blocked = [_][]const u8{&[_]u8{1}};
    try testing.expectError(MazeError.NoSolution, solveMaze(alloc, &blocked, 0, 0, 0, 0));
}
