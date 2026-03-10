//! Project Euler Problem 82: Path Sum Three Ways - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_082/sol1.py

const std = @import("std");
const testing = std.testing;

const input_file = @embedFile("problem_082_input.txt");
const test_file = @embedFile("problem_082_test_matrix.txt");

pub const Problem082Error = error{ OutOfMemory, InvalidMatrix };

const Matrix = struct {
    values: []u32,
    rows: usize,
    cols: usize,
};

fn parseMatrix(allocator: std.mem.Allocator, data: []const u8) Problem082Error!Matrix {
    var values = std.ArrayListUnmanaged(u32){};
    errdefer values.deinit(allocator);

    var rows: usize = 0;
    var cols: usize = 0;
    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| {
        var row_cols: usize = 0;
        var items = std.mem.tokenizeScalar(u8, line, ',');
        while (items.next()) |item| {
            const value = std.fmt.parseInt(u32, item, 10) catch return error.InvalidMatrix;
            try values.append(allocator, value);
            row_cols += 1;
        }
        if (row_cols == 0) continue;
        if (cols == 0) cols = row_cols else if (cols != row_cols) return error.InvalidMatrix;
        rows += 1;
    }
    if (rows == 0 or cols == 0) return error.InvalidMatrix;
    return .{ .values = try values.toOwnedSlice(allocator), .rows = rows, .cols = cols };
}

/// Returns the minimal path sum from any cell in the left column to any cell in the
/// right column, moving only up, down, and right.
/// Time complexity: O(rows * cols)
/// Space complexity: O(rows)
pub fn minimalPathSum(allocator: std.mem.Allocator, data: []const u8) Problem082Error!u32 {
    const matrix = try parseMatrix(allocator, data);
    defer allocator.free(matrix.values);

    var costs = try allocator.alloc(u32, matrix.rows);
    defer allocator.free(costs);

    for (0..matrix.rows) |row| costs[row] = matrix.values[row * matrix.cols];

    var col: usize = 1;
    while (col < matrix.cols) : (col += 1) {
        for (0..matrix.rows) |row| costs[row] += matrix.values[row * matrix.cols + col];
        for (1..matrix.rows) |row| {
            const candidate = costs[row - 1] + matrix.values[row * matrix.cols + col];
            if (candidate < costs[row]) costs[row] = candidate;
        }
        var row = matrix.rows - 1;
        while (row > 0) : (row -= 1) {
            const candidate = costs[row] + matrix.values[(row - 1) * matrix.cols + col];
            if (candidate < costs[row - 1]) costs[row - 1] = candidate;
        }
    }

    var best = costs[0];
    for (costs[1..]) |cost| best = @min(best, cost);
    return best;
}

pub fn solution(allocator: std.mem.Allocator) Problem082Error!u32 {
    return minimalPathSum(allocator, input_file);
}

test "problem 082: python reference" {
    try testing.expectEqual(@as(u32, 260324), try solution(testing.allocator));
}

test "problem 082: sample and invalid matrix" {
    try testing.expectEqual(@as(u32, 994), try minimalPathSum(testing.allocator, test_file));
    try testing.expectError(error.InvalidMatrix, minimalPathSum(testing.allocator, "1,2\n3\n"));
}
