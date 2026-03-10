//! Project Euler Problem 81: Path Sum Two Ways - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_081/sol1.py

const std = @import("std");
const testing = std.testing;

const matrix_file = @embedFile("problem_081_matrix.txt");

pub const Problem081Error = error{ OutOfMemory, InvalidMatrix };

const Matrix = struct {
    values: []u32,
    rows: usize,
    cols: usize,
};

fn parseMatrix(allocator: std.mem.Allocator, data: []const u8) Problem081Error!Matrix {
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

/// Returns the minimal path sum from the top-left to the bottom-right corner,
/// moving only right and down.
/// Time complexity: O(rows * cols)
/// Space complexity: O(rows * cols)
pub fn minimalPathSum(allocator: std.mem.Allocator, data: []const u8) Problem081Error!u32 {
    const matrix = try parseMatrix(allocator, data);
    defer allocator.free(matrix.values);

    var dp = try allocator.alloc(u32, matrix.values.len);
    defer allocator.free(dp);

    dp[0] = matrix.values[0];
    var col: usize = 1;
    while (col < matrix.cols) : (col += 1) dp[col] = matrix.values[col] + dp[col - 1];

    var row: usize = 1;
    while (row < matrix.rows) : (row += 1) {
        dp[row * matrix.cols] = matrix.values[row * matrix.cols] + dp[(row - 1) * matrix.cols];
        col = 1;
        while (col < matrix.cols) : (col += 1) {
            const idx = row * matrix.cols + col;
            dp[idx] = matrix.values[idx] + @min(dp[idx - 1], dp[idx - matrix.cols]);
        }
    }

    return dp[dp.len - 1];
}

pub fn solution(allocator: std.mem.Allocator) Problem081Error!u32 {
    return minimalPathSum(allocator, matrix_file);
}

test "problem 081: python reference" {
    try testing.expectEqual(@as(u32, 427337), try solution(testing.allocator));
}

test "problem 081: sample and invalid matrix" {
    const sample =
        \\131,673,234,103,18
        \\201,96,342,965,150
        \\630,803,746,422,111
        \\537,699,497,121,956
        \\805,732,524,37,331
    ;
    try testing.expectEqual(@as(u32, 2427), try minimalPathSum(testing.allocator, sample));
    try testing.expectError(error.InvalidMatrix, minimalPathSum(testing.allocator, "1,2\n3\n"));
}
