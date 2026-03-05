//! Jacobi Iteration Method - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/jacobi_iteration_method.py

const std = @import("std");
const testing = std.testing;

pub const JacobiError = error{
    NonSquareMatrix,
    InvalidConstantShape,
    DimensionMismatch,
    InitialValueLengthMismatch,
    IterationsMustBePositive,
    NotStrictlyDiagonallyDominant,
    ZeroDiagonal,
    Overflow,
};

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

fn validateStrictDominance(coeff: []const f64, rows: usize, cols: usize) JacobiError!void {
    for (0..rows) |r| {
        var total: f64 = 0;
        for (0..cols) |c| {
            if (c == r) continue;
            total += coeff[idx(r, c, cols)];
        }

        const diag = coeff[idx(r, r, cols)];
        if (@abs(diag) <= 1e-12) return JacobiError.ZeroDiagonal;
        if (diag <= total) return JacobiError.NotStrictlyDiagonallyDominant;
    }
}

/// Solves Ax=b approximately using Jacobi iterations.
///
/// API note: strict diagonal dominance check mirrors Python reference behavior
/// (comparison against raw off-diagonal row sum).
///
/// Time complexity: O(iterations * n^2)
/// Space complexity: O(n)
pub fn jacobiIterationMethod(
    allocator: std.mem.Allocator,
    coefficient: []const f64,
    coeff_rows: usize,
    coeff_cols: usize,
    constant: []const f64,
    const_rows: usize,
    const_cols: usize,
    init_values: []const f64,
    iterations: usize,
) (JacobiError || std.mem.Allocator.Error)![]f64 {
    if (coeff_rows != coeff_cols) return JacobiError.NonSquareMatrix;
    if (const_cols != 1) return JacobiError.InvalidConstantShape;
    if (coeff_rows != const_rows) return JacobiError.DimensionMismatch;
    if (init_values.len != coeff_rows) return JacobiError.InitialValueLengthMismatch;
    if (iterations == 0) return JacobiError.IterationsMustBePositive;

    const coeff_count = @mulWithOverflow(coeff_rows, coeff_cols);
    if (coeff_count[1] != 0) return JacobiError.Overflow;
    if (coefficient.len != coeff_count[0]) return JacobiError.DimensionMismatch;

    const const_count = @mulWithOverflow(const_rows, const_cols);
    if (const_count[1] != 0) return JacobiError.Overflow;
    if (constant.len != const_count[0]) return JacobiError.DimensionMismatch;

    try validateStrictDominance(coefficient, coeff_rows, coeff_cols);

    const current = try allocator.dupe(f64, init_values);
    defer allocator.free(current);

    var next = try allocator.alloc(f64, coeff_rows);
    defer allocator.free(next);

    for (0..iterations) |_| {
        for (0..coeff_rows) |row| {
            const diag = coefficient[idx(row, row, coeff_cols)];
            if (@abs(diag) <= 1e-12) return JacobiError.ZeroDiagonal;

            var sum: f64 = 0;
            for (0..coeff_cols) |col| {
                if (col == row) continue;
                sum += (-1.0) * coefficient[idx(row, col, coeff_cols)] * current[col];
            }

            next[row] = (sum + constant[idx(row, 0, const_cols)]) / diag;
        }
        std.mem.copyForwards(f64, current, next);
    }

    return allocator.dupe(f64, current);
}

fn expectApproxSlice(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "jacobi iteration: python example" {
    const alloc = testing.allocator;

    const coeff = [_]f64{
        4, 1, 1,
        1, 5, 2,
        1, 2, 4,
    };
    const constant = [_]f64{ 2, -6, -4 };
    const init_vals = [_]f64{ 0.5, -0.5, -0.5 };

    const result = try jacobiIterationMethod(alloc, &coeff, 3, 3, &constant, 3, 1, &init_vals, 3);
    defer alloc.free(result);

    try expectApproxSlice(&[_]f64{ 0.909375, -1.14375, -0.7484375 }, result, 1e-9);
}

test "jacobi iteration: validation errors" {
    const alloc = testing.allocator;

    const coeff_non_square = [_]f64{ 4, 1, 1, 1, 5, 2 };
    const constant3 = [_]f64{ 2, -6, -4 };
    const init3 = [_]f64{ 0.5, -0.5, -0.5 };
    try testing.expectError(
        JacobiError.NonSquareMatrix,
        jacobiIterationMethod(alloc, &coeff_non_square, 2, 3, &constant3, 3, 1, &init3, 3),
    );

    const coeff = [_]f64{
        4, 1, 1,
        1, 5, 2,
        1, 2, 4,
    };
    const bad_constant_rows = [_]f64{ 2, -6 };
    try testing.expectError(
        JacobiError.DimensionMismatch,
        jacobiIterationMethod(alloc, &coeff, 3, 3, &bad_constant_rows, 2, 1, &init3, 3),
    );

    const bad_init = [_]f64{ 0.5, -0.5 };
    try testing.expectError(
        JacobiError.InitialValueLengthMismatch,
        jacobiIterationMethod(alloc, &coeff, 3, 3, &constant3, 3, 1, &bad_init, 3),
    );

    try testing.expectError(
        JacobiError.IterationsMustBePositive,
        jacobiIterationMethod(alloc, &coeff, 3, 3, &constant3, 3, 1, &init3, 0),
    );

    try testing.expectError(
        JacobiError.InvalidConstantShape,
        jacobiIterationMethod(alloc, &coeff, 3, 3, &[_]f64{ 2, -6, -4, 1, 1, 1 }, 3, 2, &init3, 3),
    );
}

test "jacobi iteration: diagonal dominance and edge cases" {
    const alloc = testing.allocator;

    const non_dominant = [_]f64{
        4, 1, 1,
        1, 5, 2,
        1, 2, 3,
    };
    const constant3 = [_]f64{ 2, -6, -4 };
    const init3 = [_]f64{ 0.5, -0.5, -0.5 };
    try testing.expectError(
        JacobiError.NotStrictlyDiagonallyDominant,
        jacobiIterationMethod(alloc, &non_dominant, 3, 3, &constant3, 3, 1, &init3, 3),
    );

    const single = [_]f64{2};
    const single_const = [_]f64{8};
    const single_init = [_]f64{0};
    const single_result = try jacobiIterationMethod(alloc, &single, 1, 1, &single_const, 1, 1, &single_init, 1);
    defer alloc.free(single_result);
    try expectApproxSlice(&[_]f64{4}, single_result, 1e-12);
}
