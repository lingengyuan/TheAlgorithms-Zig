//! N-Queens Problem - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/n_queens.py

const std = @import("std");
const testing = std.testing;

/// Returns true if placing a queen at (row, col) is safe given queens placed in
/// previous rows (queens[i] = column of queen in row i).
fn isSafe(queens: []const usize, row: usize, col: usize) bool {
    for (0..row) |r| {
        const c = queens[r];
        if (c == col) return false;
        const dr = row - r;
        const dc = if (col > c) col - c else c - col;
        if (dr == dc) return false;
    }
    return true;
}

fn solve(
    allocator: std.mem.Allocator,
    queens: []usize,
    row: usize,
    n: usize,
    count: *usize,
    solutions: ?*std.ArrayListUnmanaged([]usize),
) !void {
    if (row == n) {
        count.* += 1;
        if (solutions) |s| {
            const copy = try allocator.dupe(usize, queens);
            try s.append(allocator, copy);
        }
        return;
    }
    for (0..n) |col| {
        if (isSafe(queens, row, col)) {
            queens[row] = col;
            try solve(allocator, queens, row + 1, n, count, solutions);
        }
    }
}

/// Counts the number of solutions to the N-Queens problem.
pub fn nQueensCount(allocator: std.mem.Allocator, n: usize) !usize {
    const queens = try allocator.alloc(usize, n);
    defer allocator.free(queens);
    var count: usize = 0;
    try solve(allocator, queens, 0, n, &count, null);
    return count;
}

/// Returns all solutions; each solution is a slice of column positions per row.
/// Caller must free each inner slice and call result.deinit().
pub fn nQueensSolutions(
    allocator: std.mem.Allocator,
    n: usize,
    result: *std.ArrayListUnmanaged([]usize),
) !void {
    const queens = try allocator.alloc(usize, n);
    defer allocator.free(queens);
    var count: usize = 0;
    try solve(allocator, queens, 0, n, &count, result);
}

test "n-queens: solution counts" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try nQueensCount(alloc, 1));
    try testing.expectEqual(@as(usize, 0), try nQueensCount(alloc, 2));
    try testing.expectEqual(@as(usize, 0), try nQueensCount(alloc, 3));
    try testing.expectEqual(@as(usize, 2), try nQueensCount(alloc, 4));
    try testing.expectEqual(@as(usize, 10), try nQueensCount(alloc, 5));
    try testing.expectEqual(@as(usize, 92), try nQueensCount(alloc, 8));
}

test "n-queens: 4x4 solutions content" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]usize){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try nQueensSolutions(alloc, 4, &result);
    try testing.expectEqual(@as(usize, 2), result.items.len);
    // Two known solutions for 4-queens: [1,3,0,2] and [2,0,3,1]
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 3, 0, 2 }, result.items[0]);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 0, 3, 1 }, result.items[1]);
}
