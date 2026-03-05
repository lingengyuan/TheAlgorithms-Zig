//! Gaussian Elimination with Partial Pivoting - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/gaussian_elimination_pivoting.py

const std = @import("std");
const testing = std.testing;

pub const GaussianPivotingError = error{
    MatrixNotSquare,
    SingularMatrix,
    DimensionMismatch,
    Overflow,
};

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

/// Solves linear system from augmented matrix (N x (N+1)).
///
/// Time complexity: O(n^3)
/// Space complexity: O(n^2)
pub fn solveLinearSystem(
    allocator: std.mem.Allocator,
    matrix: []const f64,
    rows: usize,
    cols: usize,
) (GaussianPivotingError || std.mem.Allocator.Error)![]f64 {
    const expected_cols = @addWithOverflow(rows, 1);
    if (expected_cols[1] != 0) return GaussianPivotingError.Overflow;
    if (cols != expected_cols[0]) return GaussianPivotingError.MatrixNotSquare;

    const count = @mulWithOverflow(rows, cols);
    if (count[1] != 0) return GaussianPivotingError.Overflow;
    if (matrix.len != count[0]) return GaussianPivotingError.DimensionMismatch;

    var ab = try allocator.dupe(f64, matrix);
    defer allocator.free(ab);

    const eps = 1e-8;

    for (0..rows) |column_num| {
        var pivot_row = column_num;
        for (column_num..rows) |i| {
            if (@abs(ab[idx(i, column_num, cols)]) > @abs(ab[idx(pivot_row, column_num, cols)])) {
                pivot_row = i;
            }
        }

        if (pivot_row != column_num) {
            for (0..cols) |j| {
                std.mem.swap(
                    f64,
                    &ab[idx(column_num, j, cols)],
                    &ab[idx(pivot_row, j, cols)],
                );
            }
        }

        if (@abs(ab[idx(column_num, column_num, cols)]) < eps) {
            return GaussianPivotingError.SingularMatrix;
        }

        if (column_num != 0) {
            for (column_num..rows) |i| {
                const denom = ab[idx(column_num - 1, column_num - 1, cols)];
                if (@abs(denom) < eps) return GaussianPivotingError.SingularMatrix;
                const factor = ab[idx(i, column_num - 1, cols)] / denom;
                for (0..cols) |j| {
                    ab[idx(i, j, cols)] -= factor * ab[idx(column_num - 1, j, cols)];
                }
            }
        }
    }

    const x = try allocator.alloc(f64, rows);
    errdefer allocator.free(x);

    var row = rows;
    while (row > 0) {
        row -= 1;
        const diagonal = ab[idx(row, row, cols)];
        if (@abs(diagonal) < eps) return GaussianPivotingError.SingularMatrix;

        const value = ab[idx(row, cols - 1, cols)] / diagonal;
        x[row] = value;

        var i = row;
        while (i > 0) {
            i -= 1;
            ab[idx(i, cols - 1, cols)] -= ab[idx(i, row, cols)] * value;
        }
    }

    return x;
}

test "gaussian elimination pivoting: python example" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        2,  1,  -1, 8,
        -3, -1, 2,  -11,
        -2, 1,  2,  -3,
    };
    const solution = try solveLinearSystem(alloc, &matrix, 3, 4);
    defer alloc.free(solution);

    try testing.expectApproxEqAbs(@as(f64, 2.0), solution[0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.0), solution[1], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -1.0), solution[2], 1e-9);
}

test "gaussian elimination pivoting: validation" {
    const alloc = testing.allocator;

    try testing.expectError(
        GaussianPivotingError.MatrixNotSquare,
        solveLinearSystem(alloc, &[_]f64{ 0, 0, 0 }, 1, 3),
    );

    const singular = [_]f64{
        0, 0, 0,
        0, 0, 0,
    };
    try testing.expectError(
        GaussianPivotingError.SingularMatrix,
        solveLinearSystem(alloc, &singular, 2, 3),
    );
}

test "gaussian elimination pivoting: extreme larger system consistency" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        5,  -5, -3, 4,  -11,
        1,  -4, 6,  -4, -10,
        -2, -5, 4,  -5, -12,
        -3, -3, 5,  -5, 8,
    };
    const solution = try solveLinearSystem(alloc, &matrix, 4, 5);
    defer alloc.free(solution);

    for (0..4) |r| {
        var lhs: f64 = 0;
        for (0..4) |c| {
            lhs += matrix[idx(r, c, 5)] * solution[c];
        }
        const rhs = matrix[idx(r, 4, 5)];
        try testing.expectApproxEqAbs(rhs, lhs, 1e-6);
    }
}
