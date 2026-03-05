//! Matrix Inversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/matrix_inversion.py

const std = @import("std");
const testing = std.testing;

pub const MatrixInversionError = error{ NonSquareMatrix, DimensionMismatch, SingularMatrix, Overflow };

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

/// Inverts a square matrix using Gauss-Jordan elimination with partial pivoting.
/// Matrices are provided as flat row-major slices.
///
/// Time complexity: O(n^3)
/// Space complexity: O(n^2)
pub fn invertMatrix(
    allocator: std.mem.Allocator,
    matrix: []const f64,
    rows: usize,
    cols: usize,
) (MatrixInversionError || std.mem.Allocator.Error)![]f64 {
    if (rows != cols) return MatrixInversionError.NonSquareMatrix;

    const count = @mulWithOverflow(rows, cols);
    if (count[1] != 0) return MatrixInversionError.Overflow;
    if (matrix.len != count[0]) return MatrixInversionError.DimensionMismatch;

    const aug_cols_pair = @mulWithOverflow(cols, @as(usize, 2));
    if (aug_cols_pair[1] != 0) return MatrixInversionError.Overflow;
    const aug_cols = aug_cols_pair[0];

    const aug_count = @mulWithOverflow(rows, aug_cols);
    if (aug_count[1] != 0) return MatrixInversionError.Overflow;

    var augmented = try allocator.alloc(f64, aug_count[0]);
    defer allocator.free(augmented);

    for (0..rows) |r| {
        for (0..cols) |c| {
            augmented[idx(r, c, aug_cols)] = matrix[idx(r, c, cols)];
        }
        for (0..cols) |c| {
            augmented[idx(r, cols + c, aug_cols)] = if (r == c) 1.0 else 0.0;
        }
    }

    const eps = 1e-12;

    for (0..rows) |pivot_col| {
        var pivot_row = pivot_col;
        var best_abs = @abs(augmented[idx(pivot_col, pivot_col, aug_cols)]);

        var r = pivot_col + 1;
        while (r < rows) : (r += 1) {
            const candidate = @abs(augmented[idx(r, pivot_col, aug_cols)]);
            if (candidate > best_abs) {
                best_abs = candidate;
                pivot_row = r;
            }
        }

        if (best_abs <= eps) return MatrixInversionError.SingularMatrix;

        if (pivot_row != pivot_col) {
            for (0..aug_cols) |c| {
                std.mem.swap(
                    f64,
                    &augmented[idx(pivot_row, c, aug_cols)],
                    &augmented[idx(pivot_col, c, aug_cols)],
                );
            }
        }

        const pivot = augmented[idx(pivot_col, pivot_col, aug_cols)];
        for (0..aug_cols) |c| {
            augmented[idx(pivot_col, c, aug_cols)] /= pivot;
        }

        for (0..rows) |row| {
            if (row == pivot_col) continue;

            const factor = augmented[idx(row, pivot_col, aug_cols)];
            if (@abs(factor) <= eps) continue;

            for (0..aug_cols) |c| {
                augmented[idx(row, c, aug_cols)] -= factor * augmented[idx(pivot_col, c, aug_cols)];
            }
        }
    }

    const inverse = try allocator.alloc(f64, count[0]);
    errdefer allocator.free(inverse);

    for (0..rows) |r| {
        for (0..cols) |c| {
            inverse[idx(r, c, cols)] = augmented[idx(r, cols + c, aug_cols)];
        }
    }

    return inverse;
}

fn expectApproxSlice(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "matrix inversion: python example" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        4.0, 7.0,
        2.0, 6.0,
    };

    const inverse = try invertMatrix(alloc, &matrix, 2, 2);
    defer alloc.free(inverse);

    const expected = [_]f64{
        0.6,  -0.7,
        -0.2, 0.4,
    };
    try expectApproxSlice(&expected, inverse, 1e-9);
}

test "matrix inversion: identity and 1x1" {
    const alloc = testing.allocator;

    const identity3 = [_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };
    const inv_identity = try invertMatrix(alloc, &identity3, 3, 3);
    defer alloc.free(inv_identity);
    try expectApproxSlice(&identity3, inv_identity, 1e-12);

    const single = [_]f64{2};
    const inv_single = try invertMatrix(alloc, &single, 1, 1);
    defer alloc.free(inv_single);
    try expectApproxSlice(&[_]f64{0.5}, inv_single, 1e-12);
}

test "matrix inversion: singular and validation errors" {
    const alloc = testing.allocator;

    const singular = [_]f64{
        1, 2,
        0, 0,
    };
    try testing.expectError(MatrixInversionError.SingularMatrix, invertMatrix(alloc, &singular, 2, 2));

    const non_square = [_]f64{ 1, 2, 3, 4, 5, 6 };
    try testing.expectError(MatrixInversionError.NonSquareMatrix, invertMatrix(alloc, &non_square, 2, 3));

    const bad_len = [_]f64{ 1, 2, 3 };
    try testing.expectError(MatrixInversionError.DimensionMismatch, invertMatrix(alloc, &bad_len, 2, 2));
}
