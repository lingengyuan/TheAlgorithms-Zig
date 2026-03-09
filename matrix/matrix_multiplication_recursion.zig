//! Matrix Multiplication Recursion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/matrix_multiplication_recursion.py

const std = @import("std");
const testing = std.testing;

pub const MatrixMultiplicationRecursionError = error{InvalidMatrixDimensions};

pub fn freeMatrix(allocator: std.mem.Allocator, matrix: [][]i64) void {
    for (matrix) |row| allocator.free(row);
    allocator.free(matrix);
}

fn allocMatrix(allocator: std.mem.Allocator, rows: usize, cols: usize) ![][]i64 {
    const matrix = try allocator.alloc([]i64, rows);
    errdefer allocator.free(matrix);
    var built: usize = 0;
    errdefer {
        for (matrix[0..built]) |row| allocator.free(row);
    }
    for (0..rows) |r| {
        matrix[r] = try allocator.alloc(i64, cols);
        @memset(matrix[r], 0);
        built += 1;
    }
    return matrix;
}

/// Returns true when the matrix is square.
pub fn isSquare(matrix: []const []const i64) bool {
    const len_matrix = matrix.len;
    for (matrix) |row| {
        if (row.len != len_matrix) return false;
    }
    return true;
}

/// Iterative matrix multiplication helper.
pub fn matrixMultiply(allocator: std.mem.Allocator, matrix_a: []const []const i64, matrix_b: []const []const i64) (MatrixMultiplicationRecursionError || std.mem.Allocator.Error)![][]i64 {
    if (matrix_a.len == 0 or matrix_b.len == 0) return allocator.alloc([]i64, 0);
    if (!isSquare(matrix_a) or !isSquare(matrix_b) or matrix_a.len != matrix_b.len) {
        return error.InvalidMatrixDimensions;
    }

    const n = matrix_a.len;
    const result = try allocMatrix(allocator, n, n);
    errdefer freeMatrix(allocator, result);
    for (0..n) |i| {
        for (0..n) |j| {
            for (0..n) |k| {
                result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    return result;
}

/// Recursive square-matrix multiplication.
pub fn matrixMultiplyRecursive(allocator: std.mem.Allocator, matrix_a: []const []const i64, matrix_b: []const []const i64) (MatrixMultiplicationRecursionError || std.mem.Allocator.Error)![][]i64 {
    if (matrix_a.len == 0 and matrix_b.len == 0) return allocator.alloc([]i64, 0);
    if (matrix_a.len == 0 or matrix_b.len == 0 or !isSquare(matrix_a) or !isSquare(matrix_b) or matrix_a.len != matrix_b.len) {
        return error.InvalidMatrixDimensions;
    }

    const n = matrix_a.len;
    const result = try allocMatrix(allocator, n, n);
    errdefer freeMatrix(allocator, result);

    const Context = struct {
        fn multiply(i: usize, j: usize, k: usize, a: []const []const i64, b: []const []const i64, out: [][]i64) void {
            if (i >= a.len) return;
            if (j >= b[0].len) return multiply(i + 1, 0, 0, a, b, out);
            if (k >= b.len) return multiply(i, j + 1, 0, a, b, out);
            out[i][j] += a[i][k] * b[k][j];
            return multiply(i, j, k + 1, a, b, out);
        }
    };
    Context.multiply(0, 0, 0, matrix_a, matrix_b, result);
    return result;
}

fn expectMatrixEq(expected: []const []const i64, actual: []const []const i64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_row, actual_row| {
        try testing.expectEqualSlices(i64, expected_row, actual_row);
    }
}

test "matrix multiplication recursion: python reference examples" {
    const alloc = testing.allocator;
    const matrix_1_to_4 = [_][]const i64{ &[_]i64{ 1, 2 }, &[_]i64{ 3, 4 } };
    const matrix_5_to_8 = [_][]const i64{ &[_]i64{ 5, 6 }, &[_]i64{ 7, 8 } };

    const product = try matrixMultiplyRecursive(alloc, matrix_1_to_4[0..], matrix_5_to_8[0..]);
    defer freeMatrix(alloc, product);
    try expectMatrixEq(&[_][]const i64{ &[_]i64{ 19, 22 }, &[_]i64{ 43, 50 } }, product);
}

test "matrix multiplication recursion: edge cases" {
    const alloc = testing.allocator;
    const empty = try matrixMultiplyRecursive(alloc, &[_][]const i64{}, &[_][]const i64{});
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const matrix_count_up = [_][]const i64{
        &[_]i64{ 1, 2, 3, 4 },
        &[_]i64{ 5, 6, 7, 8 },
        &[_]i64{ 9, 10, 11, 12 },
        &[_]i64{ 13, 14, 15, 16 },
    };
    const matrix_unordered = [_][]const i64{
        &[_]i64{ 5, 8, 1, 2 },
        &[_]i64{ 6, 7, 3, 0 },
        &[_]i64{ 4, 5, 9, 1 },
        &[_]i64{ 2, 6, 10, 14 },
    };
    const product = try matrixMultiplyRecursive(alloc, matrix_count_up[0..], matrix_unordered[0..]);
    defer freeMatrix(alloc, product);
    try expectMatrixEq(
        &[_][]const i64{
            &[_]i64{ 37, 61, 74, 61 },
            &[_]i64{ 105, 165, 166, 129 },
            &[_]i64{ 173, 269, 258, 197 },
            &[_]i64{ 241, 373, 350, 265 },
        },
        product,
    );

    const matrix_1_to_4 = [_][]const i64{ &[_]i64{ 1, 2 }, &[_]i64{ 3, 4 } };
    const invalid = [_][]const i64{ &[_]i64{ 5, 6 }, &[_]i64{ 7, 8, 9 } };
    try testing.expectError(error.InvalidMatrixDimensions, matrixMultiplyRecursive(alloc, matrix_1_to_4[0..], invalid[0..]));
}
