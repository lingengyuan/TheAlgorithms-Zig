//! Matrix Operation Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/matrix_operation.py

const std = @import("std");
const testing = std.testing;

pub const MatrixOperationError = error{
    InvalidMatrix,
    DimensionMismatch,
    NonSquareMatrix,
    OutOfBounds,
    SingularMatrix,
};

fn shape(matrix: []const []const f64) MatrixOperationError![2]usize {
    if (matrix.len == 0 or matrix[0].len == 0) return error.InvalidMatrix;
    const cols = matrix[0].len;
    for (matrix) |row| {
        if (row.len != cols) return error.InvalidMatrix;
    }
    return .{ matrix.len, cols };
}

fn allocMatrix(allocator: std.mem.Allocator, rows: usize, cols: usize) ![][]f64 {
    const matrix = try allocator.alloc([]f64, rows);
    errdefer allocator.free(matrix);
    var built: usize = 0;
    errdefer {
        for (matrix[0..built]) |row| allocator.free(row);
    }
    for (0..rows) |r| {
        matrix[r] = try allocator.alloc(f64, cols);
        @memset(matrix[r], 0.0);
        built += 1;
    }
    return matrix;
}

pub fn freeMatrix(allocator: std.mem.Allocator, matrix: [][]f64) void {
    for (matrix) |row| allocator.free(row);
    allocator.free(matrix);
}

/// Adds matrices of the same shape element-wise.
/// Caller owns the returned matrix.
pub fn add(allocator: std.mem.Allocator, matrices: []const []const []const f64) (MatrixOperationError || std.mem.Allocator.Error)![][]f64 {
    if (matrices.len == 0) return error.InvalidMatrix;
    const first_shape = try shape(matrices[0]);
    for (matrices[1..]) |matrix| {
        const current_shape = try shape(matrix);
        if (current_shape[0] != first_shape[0] or current_shape[1] != first_shape[1]) {
            return error.DimensionMismatch;
        }
    }

    const result = try allocMatrix(allocator, first_shape[0], first_shape[1]);
    errdefer freeMatrix(allocator, result);
    for (0..first_shape[0]) |r| {
        for (0..first_shape[1]) |c| {
            var sum: f64 = 0;
            for (matrices) |matrix| sum += matrix[r][c];
            result[r][c] = sum;
        }
    }
    return result;
}

/// Subtracts `matrix_b` from `matrix_a`.
/// Caller owns the returned matrix.
pub fn subtract(allocator: std.mem.Allocator, matrix_a: []const []const f64, matrix_b: []const []const f64) (MatrixOperationError || std.mem.Allocator.Error)![][]f64 {
    const shape_a = try shape(matrix_a);
    const shape_b = try shape(matrix_b);
    if (shape_a[0] != shape_b[0] or shape_a[1] != shape_b[1]) return error.DimensionMismatch;

    const result = try allocMatrix(allocator, shape_a[0], shape_a[1]);
    errdefer freeMatrix(allocator, result);
    for (0..shape_a[0]) |r| {
        for (0..shape_a[1]) |c| {
            result[r][c] = matrix_a[r][c] - matrix_b[r][c];
        }
    }
    return result;
}

/// Multiplies a matrix by a scalar.
/// Caller owns the returned matrix.
pub fn scalarMultiply(allocator: std.mem.Allocator, matrix: []const []const f64, scalar: f64) (MatrixOperationError || std.mem.Allocator.Error)![][]f64 {
    const matrix_shape = try shape(matrix);
    const result = try allocMatrix(allocator, matrix_shape[0], matrix_shape[1]);
    errdefer freeMatrix(allocator, result);
    for (0..matrix_shape[0]) |r| {
        for (0..matrix_shape[1]) |c| {
            result[r][c] = matrix[r][c] * scalar;
        }
    }
    return result;
}

/// Multiplies `matrix_a` by `matrix_b`.
/// Caller owns the returned matrix.
pub fn multiply(allocator: std.mem.Allocator, matrix_a: []const []const f64, matrix_b: []const []const f64) (MatrixOperationError || std.mem.Allocator.Error)![][]f64 {
    const shape_a = try shape(matrix_a);
    const shape_b = try shape(matrix_b);
    if (shape_a[1] != shape_b[0]) return error.DimensionMismatch;

    const result = try allocMatrix(allocator, shape_a[0], shape_b[1]);
    errdefer freeMatrix(allocator, result);
    for (0..shape_a[0]) |r| {
        for (0..shape_b[1]) |c| {
            var sum: f64 = 0;
            for (0..shape_a[1]) |k| {
                sum += matrix_a[r][k] * matrix_b[k][c];
            }
            result[r][c] = sum;
        }
    }
    return result;
}

/// Returns the identity matrix of order `n`.
/// Caller owns the returned matrix.
pub fn identity(allocator: std.mem.Allocator, n: usize) ![][]f64 {
    const result = try allocMatrix(allocator, n, n);
    for (0..n) |i| result[i][i] = 1.0;
    return result;
}

/// Returns the transpose of a matrix.
/// Caller owns the returned matrix.
pub fn transpose(allocator: std.mem.Allocator, matrix: []const []const f64) (MatrixOperationError || std.mem.Allocator.Error)![][]f64 {
    const matrix_shape = try shape(matrix);
    const result = try allocMatrix(allocator, matrix_shape[1], matrix_shape[0]);
    errdefer freeMatrix(allocator, result);
    for (0..matrix_shape[0]) |r| {
        for (0..matrix_shape[1]) |c| {
            result[c][r] = matrix[r][c];
        }
    }
    return result;
}

/// Returns the matrix minor at (`row`, `column`).
/// Caller owns the returned matrix.
pub fn minor(allocator: std.mem.Allocator, matrix: []const []const f64, row: usize, column: usize) (MatrixOperationError || std.mem.Allocator.Error)![][]f64 {
    const matrix_shape = try shape(matrix);
    if (row >= matrix_shape[0] or column >= matrix_shape[1]) return error.OutOfBounds;
    if (matrix_shape[0] <= 1 or matrix_shape[1] <= 1) return error.InvalidMatrix;

    const result = try allocMatrix(allocator, matrix_shape[0] - 1, matrix_shape[1] - 1);
    errdefer freeMatrix(allocator, result);

    var rr: usize = 0;
    for (0..matrix_shape[0]) |r| {
        if (r == row) continue;
        var cc: usize = 0;
        for (0..matrix_shape[1]) |c| {
            if (c == column) continue;
            result[rr][cc] = matrix[r][c];
            cc += 1;
        }
        rr += 1;
    }
    return result;
}

/// Returns the determinant of a square matrix.
pub fn determinant(allocator: std.mem.Allocator, matrix: []const []const f64) (MatrixOperationError || std.mem.Allocator.Error)!f64 {
    const matrix_shape = try shape(matrix);
    if (matrix_shape[0] != matrix_shape[1]) return error.NonSquareMatrix;

    if (matrix_shape[0] == 1) return matrix[0][0];
    if (matrix_shape[0] == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
    }

    var det: f64 = 0;
    for (0..matrix_shape[1]) |c| {
        const sub = try minor(allocator, matrix, 0, c);
        defer freeMatrix(allocator, sub);
        const sign: f64 = if (c % 2 == 0) 1.0 else -1.0;
        det += sign * matrix[0][c] * try determinant(allocator, sub);
    }
    return det;
}

/// Returns the inverse matrix, or `null` if singular.
/// Caller owns the returned matrix when present.
pub fn inverse(allocator: std.mem.Allocator, matrix: []const []const f64) (MatrixOperationError || std.mem.Allocator.Error)!?[][]f64 {
    const matrix_shape = try shape(matrix);
    if (matrix_shape[0] != matrix_shape[1]) return error.NonSquareMatrix;

    const det = try determinant(allocator, matrix);
    if (det == 0) return null;
    if (matrix_shape[0] == 1) {
        const result = try allocMatrix(allocator, 1, 1);
        result[0][0] = 1.0 / det;
        return result;
    }

    const cofactors = try allocMatrix(allocator, matrix_shape[0], matrix_shape[1]);
    errdefer freeMatrix(allocator, cofactors);
    for (0..matrix_shape[0]) |r| {
        for (0..matrix_shape[1]) |c| {
            const sub = try minor(allocator, matrix, r, c);
            defer freeMatrix(allocator, sub);
            const sign: f64 = if ((r + c) % 2 == 0) 1.0 else -1.0;
            cofactors[r][c] = sign * try determinant(allocator, sub);
        }
    }

    const adjugate = try transpose(allocator, cofactors);
    defer freeMatrix(allocator, cofactors);
    defer freeMatrix(allocator, adjugate);
    return try scalarMultiply(allocator, adjugate, 1.0 / det);
}

fn expectMatrixApproxEq(expected: []const []const f64, actual: []const []const f64, tol: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_row, actual_row| {
        try testing.expectEqual(expected_row.len, actual_row.len);
        for (expected_row, actual_row) |e, a| {
            try testing.expectApproxEqAbs(e, a, tol);
        }
    }
}

test "matrix operation: python reference examples" {
    const alloc = testing.allocator;
    const a = [_][]const f64{ &[_]f64{ 1, 2 }, &[_]f64{ 3, 4 } };
    const b = [_][]const f64{ &[_]f64{ 2, 3 }, &[_]f64{ 4, 5 } };
    const c = [_][]const f64{ &[_]f64{ 3, 5 }, &[_]f64{ 5, 7 } };
    const extra = [_][]const f64{ &[_]f64{ 3, 7 }, &[_]f64{ 3, 4 } };
    const multiplier = [_][]const f64{ &[_]f64{ 5, 5 }, &[_]f64{ 7, 5 } };
    const add_base = [_][]const f64{ &[_]f64{ 1, 2 }, &[_]f64{ 4, 5 } };

    const sum = try add(alloc, &[_][]const []const f64{ a[0..], b[0..] });
    defer freeMatrix(alloc, sum);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ 3, 5 }, &[_]f64{ 7, 9 } }, sum, 1e-12);

    const sum3 = try add(alloc, &[_][]const []const f64{ add_base[0..], extra[0..], c[0..] });
    defer freeMatrix(alloc, sum3);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ 7, 14 }, &[_]f64{ 12, 16 } }, sum3, 1e-12);

    const diff = try subtract(alloc, a[0..], b[0..]);
    defer freeMatrix(alloc, diff);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ -1, -1 }, &[_]f64{ -1, -1 } }, diff, 1e-12);

    const prod = try multiply(alloc, a[0..], multiplier[0..]);
    defer freeMatrix(alloc, prod);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ 19, 15 }, &[_]f64{ 43, 35 } }, prod, 1e-12);

    const inv = (try inverse(alloc, a[0..])).?;
    defer freeMatrix(alloc, inv);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ -2.0, 1.0 }, &[_]f64{ 1.5, -0.5 } }, inv, 1e-12);
}

test "matrix operation: edge cases" {
    const alloc = testing.allocator;
    const simple = [_][]const f64{ &[_]f64{ 1, 2 }, &[_]f64{ 3, 4 } };
    const invalid_left = [_][]const f64{&[_]f64{ 1, 2, 3 }};
    const invalid_right = [_][]const f64{ &[_]f64{2}, &[_]f64{3} };

    const id = try identity(alloc, 3);
    defer freeMatrix(alloc, id);
    try expectMatrixApproxEq(
        &[_][]const f64{
            &[_]f64{ 1, 0, 0 },
            &[_]f64{ 0, 1, 0 },
            &[_]f64{ 0, 0, 1 },
        },
        id,
        1e-12,
    );

    const trans = try transpose(alloc, simple[0..]);
    defer freeMatrix(alloc, trans);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ 1, 3 }, &[_]f64{ 2, 4 } }, trans, 1e-12);

    const sub = try minor(alloc, simple[0..], 1, 1);
    defer freeMatrix(alloc, sub);
    try expectMatrixApproxEq(&[_][]const f64{&[_]f64{1}}, sub, 1e-12);

    try testing.expectError(error.DimensionMismatch, multiply(alloc, invalid_left[0..], invalid_right[0..]));
}
