//! Rayleigh Quotient - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/rayleigh_quotient.py

const std = @import("std");
const testing = std.testing;

pub const RayleighError = error{ NonSquareMatrix, DimensionMismatch, ZeroVector, Overflow };

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

/// Checks whether a real matrix is symmetric.
pub fn isSymmetric(matrix: []const f64, rows: usize, cols: usize) bool {
    if (rows != cols) return false;
    if (matrix.len != rows * cols) return false;

    for (0..rows) |r| {
        for (r + 1..cols) |c| {
            if (matrix[idx(r, c, cols)] != matrix[idx(c, r, cols)]) return false;
        }
    }
    return true;
}

/// Computes Rayleigh quotient: (v^T A v) / (v^T v).
///
/// Time complexity: O(n^2)
/// Space complexity: O(1)
pub fn rayleighQuotient(
    matrix: []const f64,
    rows: usize,
    cols: usize,
    vector: []const f64,
) RayleighError!f64 {
    if (rows != cols) return RayleighError.NonSquareMatrix;

    const count = @mulWithOverflow(rows, cols);
    if (count[1] != 0) return RayleighError.Overflow;
    if (matrix.len != count[0]) return RayleighError.DimensionMismatch;
    if (vector.len != rows) return RayleighError.DimensionMismatch;

    var denominator: f64 = 0;
    for (vector) |v| denominator += v * v;
    if (@abs(denominator) <= 1e-12) return RayleighError.ZeroVector;

    var numerator: f64 = 0;
    for (0..rows) |i| {
        for (0..cols) |j| {
            numerator += vector[i] * matrix[idx(i, j, cols)] * vector[j];
        }
    }

    return numerator / denominator;
}

test "rayleigh quotient: python real example" {
    const a = [_]f64{
        1, 2,  4,
        2, 3,  -1,
        4, -1, 1,
    };
    const v = [_]f64{ 1, 2, 3 };

    try testing.expect(isSymmetric(&a, 3, 3));
    const q = try rayleighQuotient(&a, 3, 3, &v);
    try testing.expectApproxEqAbs(@as(f64, 3.0), q, 1e-12);
}

test "rayleigh quotient: validation" {
    const bad = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const v2 = [_]f64{ 1, 2 };
    try testing.expectError(RayleighError.NonSquareMatrix, rayleighQuotient(&bad, 2, 3, &v2));

    const a = [_]f64{ 1, 2, 2, 1 };
    try testing.expectError(RayleighError.DimensionMismatch, rayleighQuotient(&a, 2, 2, &[_]f64{1}));
    try testing.expectError(RayleighError.ZeroVector, rayleighQuotient(&a, 2, 2, &[_]f64{ 0, 0 }));
}

test "rayleigh quotient: symmetry helper and extreme values" {
    const non_sym = [_]f64{ 1, 2, 3, 4 };
    try testing.expect(!isSymmetric(&non_sym, 2, 2));

    const a = [_]f64{
        1e6, 0,
        0,   2e6,
    };
    const v = [_]f64{ 1e3, -1e3 };
    const q = try rayleighQuotient(&a, 2, 2, &v);
    try testing.expect(q > 1.4e6 and q < 1.6e6);
}
