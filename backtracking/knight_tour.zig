//! Knight Tour - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/knight_tour.py

const std = @import("std");
const testing = std.testing;

pub const KnightTourError = error{ InvalidBoardSize, NoTour };

const Move = struct {
    dr: isize,
    dc: isize,
};

const knight_moves = [_]Move{
    .{ .dr = 1, .dc = 2 },
    .{ .dr = -1, .dc = 2 },
    .{ .dr = 1, .dc = -2 },
    .{ .dr = -1, .dc = -2 },
    .{ .dr = 2, .dc = 1 },
    .{ .dr = 2, .dc = -1 },
    .{ .dr = -2, .dc = 1 },
    .{ .dr = -2, .dc = -1 },
};

fn indexOf(n: usize, row: usize, col: usize) usize {
    return row * n + col;
}

fn isValidPosition(row: isize, col: isize, n: usize) bool {
    if (row < 0 or col < 0) return false;
    const r: usize = @intCast(row);
    const c: usize = @intCast(col);
    return r < n and c < n;
}

/// Returns all valid knight moves from a given position on an `n x n` board.
///
/// Time complexity: O(1)
/// Space complexity: O(1) auxiliary (O(k) output, k <= 8)
pub fn getValidPositions(
    allocator: std.mem.Allocator,
    position: [2]usize,
    n: usize,
) std.mem.Allocator.Error![][2]usize {
    var out = std.ArrayListUnmanaged([2]usize){};
    defer out.deinit(allocator);

    const row: isize = @intCast(position[0]);
    const col: isize = @intCast(position[1]);
    for (knight_moves) |move| {
        const nr = row + move.dr;
        const nc = col + move.dc;
        if (!isValidPosition(nr, nc, n)) continue;
        try out.append(allocator, .{ @intCast(nr), @intCast(nc) });
    }

    return out.toOwnedSlice(allocator);
}

/// Returns `true` when all board cells are non-zero.
pub fn isComplete(board: []const usize) bool {
    for (board) |cell| {
        if (cell == 0) return false;
    }
    return true;
}

fn onwardDegree(board: []const usize, n: usize, row: isize, col: isize) u8 {
    var degree: u8 = 0;
    for (knight_moves) |move| {
        const nr = row + move.dr;
        const nc = col + move.dc;
        if (!isValidPosition(nr, nc, n)) continue;
        const r: usize = @intCast(nr);
        const c: usize = @intCast(nc);
        if (board[indexOf(n, r, c)] == 0) degree += 1;
    }
    return degree;
}

const Candidate = struct {
    row: isize,
    col: isize,
    degree: u8,
};

fn sortCandidates(candidates: *[8]Candidate, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        var best = i;
        var j = i + 1;
        while (j < len) : (j += 1) {
            if (candidates[j].degree < candidates[best].degree) best = j;
        }
        if (best != i) std.mem.swap(Candidate, &candidates[i], &candidates[best]);
    }
}

fn solveTour(board: []usize, n: usize, row: isize, col: isize, step: usize) bool {
    const total = n * n;
    if (step > total) return true;

    var candidates: [8]Candidate = undefined;
    var count: usize = 0;

    for (knight_moves) |move| {
        const nr = row + move.dr;
        const nc = col + move.dc;
        if (!isValidPosition(nr, nc, n)) continue;

        const r: usize = @intCast(nr);
        const c: usize = @intCast(nc);
        if (board[indexOf(n, r, c)] != 0) continue;

        candidates[count] = .{
            .row = nr,
            .col = nc,
            .degree = onwardDegree(board, n, nr, nc),
        };
        count += 1;
    }

    sortCandidates(&candidates, count);

    for (candidates[0..count]) |candidate| {
        const r: usize = @intCast(candidate.row);
        const c: usize = @intCast(candidate.col);
        const idx = indexOf(n, r, c);
        board[idx] = step;
        if (solveTour(board, n, candidate.row, candidate.col, step + 1)) return true;
        board[idx] = 0;
    }

    return false;
}

/// Finds an open knight tour for a board of size `n`.
/// Returns a flattened board where cells contain visit order starting at `1`.
///
/// Time complexity: worst-case exponential backtracking.
/// Space complexity: O(n^2) board + O(n^2) recursion depth.
pub fn openKnightTour(
    allocator: std.mem.Allocator,
    n: usize,
) (KnightTourError || std.mem.Allocator.Error)![]usize {
    if (n == 0) return KnightTourError.InvalidBoardSize;

    const board = try allocator.alloc(usize, n * n);
    @memset(board, 0);

    if (n == 1) {
        board[0] = 1;
        return board;
    }

    for (0..n) |r| {
        for (0..n) |c| {
            @memset(board, 0);
            board[indexOf(n, r, c)] = 1;
            if (solveTour(board, n, @intCast(r), @intCast(c), 2)) return board;
        }
    }

    allocator.free(board);
    return KnightTourError.NoTour;
}

fn isKnightMove(from: [2]usize, to: [2]usize) bool {
    const dr = if (from[0] > to[0]) from[0] - to[0] else to[0] - from[0];
    const dc = if (from[1] > to[1]) from[1] - to[1] else to[1] - from[1];
    return (dr == 1 and dc == 2) or (dr == 2 and dc == 1);
}

fn validateTour(allocator: std.mem.Allocator, board: []const usize, n: usize) !void {
    const total = n * n;
    try testing.expectEqual(total, board.len);

    const seen = try allocator.alloc(bool, total + 1);
    defer allocator.free(seen);
    @memset(seen, false);

    const positions = try allocator.alloc([2]usize, total + 1);
    defer allocator.free(positions);

    for (board, 0..) |step, idx| {
        try testing.expect(step >= 1 and step <= total);
        try testing.expect(!seen[step]);
        seen[step] = true;
        positions[step] = .{ idx / n, idx % n };
    }

    for (1..total) |step| {
        try testing.expect(isKnightMove(positions[step], positions[step + 1]));
    }

    try testing.expect(isComplete(board));
}

test "knight tour: valid positions example" {
    const alloc = testing.allocator;
    const positions = try getValidPositions(alloc, .{ 1, 3 }, 4);
    defer alloc.free(positions);

    const expected = [_][2]usize{
        .{ 2, 1 },
        .{ 0, 1 },
        .{ 3, 2 },
    };
    try testing.expectEqualSlices([2]usize, &expected, positions);
}

test "knight tour: completion helper" {
    try testing.expect(isComplete(&[_]usize{1}));
    try testing.expect(!isComplete(&[_]usize{ 1, 2, 0 }));
}

test "knight tour: python base cases" {
    const alloc = testing.allocator;

    const one = try openKnightTour(alloc, 1);
    defer alloc.free(one);
    try testing.expectEqualSlices(usize, &[_]usize{1}, one);

    try testing.expectError(KnightTourError.NoTour, openKnightTour(alloc, 2));
}

test "knight tour: unsolved small boards and invalid size" {
    const alloc = testing.allocator;
    try testing.expectError(KnightTourError.InvalidBoardSize, openKnightTour(alloc, 0));
    try testing.expectError(KnightTourError.NoTour, openKnightTour(alloc, 3));
    try testing.expectError(KnightTourError.NoTour, openKnightTour(alloc, 4));
}

test "knight tour: solvable medium board" {
    const alloc = testing.allocator;
    const board = try openKnightTour(alloc, 5);
    defer alloc.free(board);
    try validateTour(alloc, board, 5);
}

test "knight tour: extreme larger board" {
    const alloc = testing.allocator;
    const board = try openKnightTour(alloc, 6);
    defer alloc.free(board);
    try validateTour(alloc, board, 6);
}
