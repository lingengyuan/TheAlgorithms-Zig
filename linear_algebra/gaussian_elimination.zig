//! Gaussian Elimination - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/gaussian_elimination.py

const std = @import("std");
const testing = std.testing;

pub const GaussianError = error{ NonSquareMatrix, DimensionMismatch, SingularMatrix, Overflow };

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

/// Solves Ax=b with Gaussian elimination.
///
/// API note: unlike Python reference (which returns empty array on non-square input),
/// this Zig implementation reports `error.NonSquareMatrix`.
///
/// Time complexity: O(n^3)
/// Space complexity: O(n^2)
pub fn gaussianElimination(
    allocator: std.mem.Allocator,
    coefficients: []const f64,
    rows: usize,
    cols: usize,
    vector: []const f64,
) (GaussianError || std.mem.Allocator.Error)![]f64 {
    if (rows != cols) return GaussianError.NonSquareMatrix;

    const coeff_count = @mulWithOverflow(rows, cols);
    if (coeff_count[1] != 0) return GaussianError.Overflow;
    if (coefficients.len != coeff_count[0]) return GaussianError.DimensionMismatch;
    if (vector.len != rows) return GaussianError.DimensionMismatch;

    const aug_cols_pair = @addWithOverflow(cols, @as(usize, 1));
    if (aug_cols_pair[1] != 0) return GaussianError.Overflow;
    const aug_cols = aug_cols_pair[0];

    const aug_count = @mulWithOverflow(rows, aug_cols);
    if (aug_count[1] != 0) return GaussianError.Overflow;

    var augmented = try allocator.alloc(f64, aug_count[0]);
    defer allocator.free(augmented);

    for (0..rows) |r| {
        for (0..cols) |c| {
            augmented[idx(r, c, aug_cols)] = coefficients[idx(r, c, cols)];
        }
        augmented[idx(r, cols, aug_cols)] = vector[r];
    }

    const epsilon = 1e-12;

    for (0..rows) |pivot_col| {
        // Partial pivoting for numerical stability and zero-pivot avoidance.
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

        if (best_abs <= epsilon) return GaussianError.SingularMatrix;

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

        r = pivot_col + 1;
        while (r < rows) : (r += 1) {
            const factor = augmented[idx(r, pivot_col, aug_cols)] / pivot;
            var c = pivot_col;
            while (c < aug_cols) : (c += 1) {
                augmented[idx(r, c, aug_cols)] -= factor * augmented[idx(pivot_col, c, aug_cols)];
            }
        }
    }

    const x = try allocator.alloc(f64, rows);
    errdefer allocator.free(x);

    var row_i: isize = @intCast(rows);
    while (row_i > 0) {
        row_i -= 1;
        const row: usize = @intCast(row_i);

        var rhs = augmented[idx(row, cols, aug_cols)];
        var c = row + 1;
        while (c < cols) : (c += 1) {
            rhs -= augmented[idx(row, c, aug_cols)] * x[c];
        }

        const diag = augmented[idx(row, row, aug_cols)];
        if (@abs(diag) <= epsilon) return GaussianError.SingularMatrix;
        x[row] = rhs / diag;
    }

    return x;
}

fn expectApproxSlice(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "gaussian elimination: 3x3 example" {
    const alloc = testing.allocator;

    const a = [_]f64{
        1, -4, -2,
        5, 2,  -2,
        1, -1, 0,
    };
    const b = [_]f64{ -2, -3, 4 };

    const x = try gaussianElimination(alloc, &a, 3, 3, &b);
    defer alloc.free(x);

    try expectApproxSlice(&[_]f64{ 2.3, -1.7, 5.55 }, x, 1e-9);
}

test "gaussian elimination: 2x2 example" {
    const alloc = testing.allocator;

    const a = [_]f64{
        1, 2,
        5, 2,
    };
    const b = [_]f64{ 5, 5 };

    const x = try gaussianElimination(alloc, &a, 2, 2, &b);
    defer alloc.free(x);

    try expectApproxSlice(&[_]f64{ 0.0, 2.5 }, x, 1e-9);
}

test "gaussian elimination: pivot swap case" {
    const alloc = testing.allocator;

    const a = [_]f64{
        0, 1,
        1, 1,
    };
    const b = [_]f64{ 1, 2 };

    const x = try gaussianElimination(alloc, &a, 2, 2, &b);
    defer alloc.free(x);

    try expectApproxSlice(&[_]f64{ 1.0, 1.0 }, x, 1e-9);
}

test "gaussian elimination: input validation" {
    const alloc = testing.allocator;

    const non_square = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b2 = [_]f64{ 7, 8 };
    try testing.expectError(GaussianError.NonSquareMatrix, gaussianElimination(alloc, &non_square, 2, 3, &b2));

    const square = [_]f64{ 1, 2, 3, 4 };
    const bad_vector = [_]f64{5};
    try testing.expectError(GaussianError.DimensionMismatch, gaussianElimination(alloc, &square, 2, 2, &bad_vector));
}

test "gaussian elimination: singular matrix" {
    const alloc = testing.allocator;

    const singular = [_]f64{
        1, 2,
        2, 4,
    };
    const b = [_]f64{ 3, 6 };

    try testing.expectError(GaussianError.SingularMatrix, gaussianElimination(alloc, &singular, 2, 2, &b));
}
