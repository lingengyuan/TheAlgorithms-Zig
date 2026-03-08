//! Game of Life (Numpy-Slice Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/cellular_automata/game_of_life.py

const std = @import("std");
const testing = std.testing;

pub const GameOfLifeError = error{
    InvalidCanvasSize,
    EmptyCanvas,
    NonRectangularCanvas,
};

fn freeCanvas(allocator: std.mem.Allocator, canvas: [][]bool) void {
    for (canvas) |row| allocator.free(row);
    allocator.free(canvas);
}

fn validateCanvas(canvas: []const []const bool) GameOfLifeError!usize {
    if (canvas.len == 0 or canvas[0].len == 0) return GameOfLifeError.EmptyCanvas;
    const width = canvas[0].len;
    for (canvas) |row| {
        if (row.len != width) return GameOfLifeError.NonRectangularCanvas;
    }
    return width;
}

const SliceBounds = struct {
    start: usize,
    stop: usize,
};

fn pythonSliceBounds(length: usize, start_index: isize, stop_index: isize) SliceBounds {
    const n: isize = @intCast(length);

    var start = start_index;
    if (start < 0) start += n;
    if (start < 0) start = 0;
    if (start > n) start = n;

    var stop = stop_index;
    if (stop < 0) stop += n;
    if (stop < 0) stop = 0;
    if (stop > n) stop = n;

    return .{ .start = @intCast(start), .stop = @intCast(stop) };
}

fn judgePoint(point: bool, neighbours: []const []const bool) bool {
    var alive: i32 = 0;

    for (neighbours) |row| {
        for (row) |status| {
            if (status) alive += 1;
        }
    }

    if (point) alive -= 1;

    var state = point;
    if (point) {
        if (alive < 2) {
            state = false;
        } else if (alive == 2 or alive == 3) {
            state = true;
        } else if (alive > 3) {
            state = false;
        }
    } else if (alive == 3) {
        state = true;
    }

    return state;
}

/// Creates square canvas initialized with dead cells.
/// Caller owns returned grid.
///
/// Time complexity: O(size^2)
/// Space complexity: O(size^2)
pub fn createCanvas(allocator: std.mem.Allocator, size: usize) ![][]bool {
    if (size == 0) return GameOfLifeError.InvalidCanvasSize;

    const canvas = try allocator.alloc([]bool, size);
    errdefer allocator.free(canvas);

    var built_rows: usize = 0;
    errdefer {
        for (0..built_rows) |i| allocator.free(canvas[i]);
    }

    for (0..size) |r| {
        canvas[r] = try allocator.alloc(bool, size);
        built_rows += 1;
        @memset(canvas[r], false);
    }

    return canvas;
}

/// Seeds canvas with random boolean values.
pub fn seed(canvas: [][]bool, random: *std.Random) void {
    for (canvas) |row| {
        for (row) |*cell| {
            cell.* = random.boolean();
        }
    }
}

/// Runs one simulation step with the same slicing semantics as the Python
/// reference implementation (`current[r-1:r+2, c-1:c+2]`).
/// Caller owns returned grid.
///
/// Time complexity: O(size^2)
/// Space complexity: O(size^2)
pub fn run(allocator: std.mem.Allocator, canvas: []const []const bool) ![][]bool {
    const width = try validateCanvas(canvas);
    const height = canvas.len;

    const next = try allocator.alloc([]bool, height);
    errdefer allocator.free(next);

    var built_rows: usize = 0;
    errdefer {
        for (0..built_rows) |i| allocator.free(next[i]);
    }

    for (0..height) |r| {
        next[r] = try allocator.alloc(bool, width);
        built_rows += 1;

        for (0..width) |c| {
            const row_bounds = pythonSliceBounds(height, @as(isize, @intCast(r)) - 1, @as(isize, @intCast(r)) + 2);
            const col_bounds = pythonSliceBounds(width, @as(isize, @intCast(c)) - 1, @as(isize, @intCast(c)) + 2);

            var neighbour_rows = std.ArrayListUnmanaged([]const bool){};
            defer neighbour_rows.deinit(allocator);

            var rr = row_bounds.start;
            while (rr < row_bounds.stop) : (rr += 1) {
                const stop = if (col_bounds.start <= col_bounds.stop) col_bounds.stop else col_bounds.start;
                try neighbour_rows.append(allocator, canvas[rr][col_bounds.start..stop]);
            }

            next[r][c] = judgePoint(canvas[r][c], neighbour_rows.items);
        }
    }

    return next;
}

test "game of life numpy variant: python run examples" {
    const alloc = testing.allocator;

    const b1 = [_][]const bool{
        &[_]bool{ false, true, false },
        &[_]bool{ false, true, false },
        &[_]bool{ false, true, false },
    };
    const r1 = try run(alloc, &b1);
    defer freeCanvas(alloc, r1);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false }, r1[0]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, true, true }, r1[1]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false }, r1[2]);

    const b2 = [_][]const bool{
        &[_]bool{ true, true, true },
        &[_]bool{ true, false, false },
        &[_]bool{ false, false, true },
    };
    const r2 = try run(alloc, &b2);
    defer freeCanvas(alloc, r2);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false }, r2[0]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, true }, r2[1]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false }, r2[2]);
}

test "game of life numpy variant: deterministic multi-step and boundaries" {
    const alloc = testing.allocator;

    const board = [_][]const bool{
        &[_]bool{ true, false, true, false },
        &[_]bool{ false, true, true, false },
        &[_]bool{ true, false, false, true },
        &[_]bool{ false, false, true, false },
    };

    const step1 = try run(alloc, &board);
    defer freeCanvas(alloc, step1);
    const step2 = try run(alloc, step1);
    defer freeCanvas(alloc, step2);

    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false, false }, step1[0]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, true, true }, step1[1]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false, true }, step1[2]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false, false }, step1[3]);

    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false, false }, step2[0]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, true, true }, step2[1]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, true, true }, step2[2]);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, false, false }, step2[3]);

    try testing.expectError(GameOfLifeError.InvalidCanvasSize, createCanvas(alloc, 0));
    try testing.expectError(GameOfLifeError.EmptyCanvas, run(alloc, &[_][]const bool{}));

    const non_rect = [_][]const bool{
        &[_]bool{ true, false },
        &[_]bool{true},
    };
    try testing.expectError(GameOfLifeError.NonRectangularCanvas, run(alloc, &non_rect));

    const large = try createCanvas(alloc, 128);
    defer freeCanvas(alloc, large);
    const large_next = try run(alloc, large);
    defer freeCanvas(alloc, large_next);

    for (large_next) |row| {
        for (row) |cell| {
            try testing.expectEqual(false, cell);
        }
    }
}
