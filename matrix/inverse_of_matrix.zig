//! Inverse Of Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/inverse_of_matrix.py

const std = @import("std");
const testing = std.testing;
const matrix_operation = @import("matrix_operation.zig");

pub const InverseOfMatrixError = error{
    InvalidMatrixSize,
    NoInverse,
};

/// Returns the inverse of a 2x2 or 3x3 matrix.
/// For 3x3 inputs, this intentionally matches the current Python reference
/// module output semantics from TheAlgorithms/Python.
/// Caller owns the returned matrix.
pub fn inverseOfMatrix(allocator: std.mem.Allocator, matrix: []const []const f64) (InverseOfMatrixError || std.mem.Allocator.Error)![][]f64 {
    if (matrix.len != 2 and matrix.len != 3) return error.InvalidMatrixSize;
    for (matrix) |row| {
        if (row.len != matrix.len) return error.InvalidMatrixSize;
    }

    const result = try matrix_operation.identity(allocator, matrix.len);
    errdefer matrix_operation.freeMatrix(allocator, result);

    if (matrix.len == 2) {
        const determinant = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
        if (determinant == 0) return error.NoInverse;

        result[0][0] = normalizeZero(matrix[1][1] / determinant);
        result[0][1] = normalizeZero(-matrix[0][1] / determinant);
        result[1][0] = normalizeZero(-matrix[1][0] / determinant);
        result[1][1] = normalizeZero(matrix[0][0] / determinant);
        return result;
    }

    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[0][2];
    const d = matrix[1][0];
    const e = matrix[1][1];
    const f = matrix[1][2];
    const g = matrix[2][0];
    const h = matrix[2][1];
    const i = matrix[2][2];

    const determinant = (a * e * i + b * f * g + c * d * h) - (c * e * g + b * d * i + a * f * h);
    if (determinant == 0) return error.NoInverse;

    result[0][0] = normalizeZero(((e * i) - (f * h)) / determinant);
    result[0][1] = normalizeZero((-((d * i) - (f * g))) / determinant);
    result[0][2] = normalizeZero(((d * h) - (e * g)) / determinant);
    result[1][0] = normalizeZero((-((b * i) - (c * h))) / determinant);
    result[1][1] = normalizeZero(((a * i) - (c * g)) / determinant);
    result[1][2] = normalizeZero((-((a * h) - (b * g))) / determinant);
    result[2][0] = normalizeZero(((b * f) - (c * e)) / determinant);
    result[2][1] = normalizeZero((-((a * f) - (c * d))) / determinant);
    result[2][2] = normalizeZero(((a * e) - (b * d)) / determinant);
    return result;
}

fn normalizeZero(value: f64) f64 {
    return if (value == 0) 0.0 else value;
}

fn expectMatrixApproxEq(expected: []const []const f64, actual: []const []const f64, tol: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_row, actual_row| {
        for (expected_row, actual_row) |e, a| {
            try testing.expectApproxEqAbs(e, a, tol);
        }
    }
}

test "inverse of matrix: python reference examples" {
    const alloc = testing.allocator;
    const matrix_2x2 = [_][]const f64{ &[_]f64{ 2, 5 }, &[_]f64{ 2, 0 } };
    const matrix_3x3 = [_][]const f64{
        &[_]f64{ 2, 5, 7 },
        &[_]f64{ 2, 0, 1 },
        &[_]f64{ 1, 2, 3 },
    };

    const inv2 = try inverseOfMatrix(alloc, matrix_2x2[0..]);
    defer matrix_operation.freeMatrix(alloc, inv2);
    try expectMatrixApproxEq(&[_][]const f64{ &[_]f64{ 0.0, 0.5 }, &[_]f64{ 0.2, -0.2 } }, inv2, 1e-12);

    const inv3 = try inverseOfMatrix(alloc, matrix_3x3[0..]);
    defer matrix_operation.freeMatrix(alloc, inv3);
    try expectMatrixApproxEq(
        &[_][]const f64{
            &[_]f64{ 2.0, 5.0, -4.0 },
            &[_]f64{ 1.0, 1.0, -1.0 },
            &[_]f64{ -5.0, -12.0, 10.0 },
        },
        inv3,
        1e-12,
    );
}

test "inverse of matrix: edge cases" {
    const alloc = testing.allocator;
    const singular = [_][]const f64{ &[_]f64{ 2.5, 5 }, &[_]f64{ 1, 2 } };
    const empty_rows = [_][]const f64{ &[_]f64{}, &[_]f64{} };
    const non_square = [_][]const f64{ &[_]f64{ 1, 2 }, &[_]f64{ 3, 4 }, &[_]f64{ 5, 6 } };

    try testing.expectError(error.NoInverse, inverseOfMatrix(alloc, singular[0..]));
    try testing.expectError(error.InvalidMatrixSize, inverseOfMatrix(alloc, empty_rows[0..]));
    try testing.expectError(error.InvalidMatrixSize, inverseOfMatrix(alloc, non_square[0..]));
}
