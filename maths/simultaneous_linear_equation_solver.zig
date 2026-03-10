//! Simultaneous Linear Equation Solver - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/simultaneous_linear_equation_solver.py

const std = @import("std");
const testing = std.testing;

pub const SolveError = error{
    InvalidShape,
    InvalidEquationSet,
    SingularMatrix,
    OutOfMemory,
};

/// Solves an `n x (n + 1)` augmented matrix by Gaussian elimination with partial pivoting.
/// Caller owns the returned slice.
/// Time complexity: O(n^3), Space complexity: O(n^2)
pub fn solveSimultaneous(allocator: std.mem.Allocator, equations: []const []const f64) SolveError![]f64 {
    if (equations.len == 0) return error.InvalidShape;
    const width = equations.len + 1;
    for (equations) |row| {
        if (row.len != width) return error.InvalidShape;
    }

    var has_full_row = false;
    for (equations) |row| {
        var full = true;
        for (row) |value| {
            if (!std.math.isFinite(value)) return error.InvalidEquationSet;
            if (@abs(value) <= 1e-12) full = false;
        }
        if (full) {
            has_full_row = true;
            break;
        }
    }
    if (!has_full_row) return error.InvalidEquationSet;

    const matrix = try allocator.alloc(f64, equations.len * width);
    defer allocator.free(matrix);

    for (equations, 0..) |row, row_index| {
        for (row, 0..) |value, col_index| {
            matrix[row_index * width + col_index] = value;
        }
    }

    var pivot_col: usize = 0;
    while (pivot_col < equations.len) : (pivot_col += 1) {
        var best_row = pivot_col;
        var best_abs = @abs(matrix[pivot_col * width + pivot_col]);

        var candidate = pivot_col + 1;
        while (candidate < equations.len) : (candidate += 1) {
            const magnitude = @abs(matrix[candidate * width + pivot_col]);
            if (magnitude > best_abs) {
                best_abs = magnitude;
                best_row = candidate;
            }
        }

        if (best_abs <= 1e-12) return error.SingularMatrix;
        if (best_row != pivot_col) swapRows(matrix, width, pivot_col, best_row);

        const pivot = matrix[pivot_col * width + pivot_col];
        var row = pivot_col + 1;
        while (row < equations.len) : (row += 1) {
            const factor = matrix[row * width + pivot_col] / pivot;
            matrix[row * width + pivot_col] = 0.0;
            var col = pivot_col + 1;
            while (col < width) : (col += 1) {
                matrix[row * width + col] -= factor * matrix[pivot_col * width + col];
            }
        }
    }

    const result = try allocator.alloc(f64, equations.len);
    errdefer allocator.free(result);

    var idx = equations.len;
    while (idx > 0) {
        idx -= 1;
        var value = matrix[idx * width + (width - 1)];
        var col = idx + 1;
        while (col < equations.len) : (col += 1) {
            value -= matrix[idx * width + col] * result[col];
        }
        const pivot = matrix[idx * width + idx];
        if (@abs(pivot) <= 1e-12) return error.SingularMatrix;
        result[idx] = round5(value / pivot);
    }

    return result;
}

fn swapRows(matrix: []f64, width: usize, first: usize, second: usize) void {
    var col: usize = 0;
    while (col < width) : (col += 1) {
        std.mem.swap(f64, &matrix[first * width + col], &matrix[second * width + col]);
    }
}

fn round5(value: f64) f64 {
    return @round(value * 100_000.0) / 100_000.0;
}

test "simultaneous linear equation solver: python reference examples" {
    const alloc = testing.allocator;

    const equations1 = [_][]const f64{
        &[_]f64{ 1, 2, 3 },
        &[_]f64{ 4, 5, 6 },
    };
    const solution1 = try solveSimultaneous(alloc, &equations1);
    defer alloc.free(solution1);
    try testing.expectEqualSlices(f64, &[_]f64{ -1.0, 2.0 }, solution1);

    const equations2 = [_][]const f64{
        &[_]f64{ 0, -3, 1, 7 },
        &[_]f64{ 3, 2, -1, 11 },
        &[_]f64{ 5, 1, -2, 12 },
    };
    const solution2 = try solveSimultaneous(alloc, &equations2);
    defer alloc.free(solution2);
    try testing.expectEqualSlices(f64, &[_]f64{ 6.4, 1.2, 10.6 }, solution2);
}

test "simultaneous linear equation solver: invalid shapes and invalid equation set" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidShape, solveSimultaneous(alloc, &[_][]const f64{}));

    const bad_shape = [_][]const f64{
        &[_]f64{ 1, 2, 3 },
        &[_]f64{ 1, 2 },
    };
    try testing.expectError(error.InvalidShape, solveSimultaneous(alloc, &bad_shape));

    const no_full_row = [_][]const f64{
        &[_]f64{ 0, 2, 3 },
        &[_]f64{ 4, 0, 6 },
    };
    try testing.expectError(error.InvalidEquationSet, solveSimultaneous(alloc, &no_full_row));
}

test "simultaneous linear equation solver: extreme diagonal system" {
    const alloc = testing.allocator;
    const equations = [_][]const f64{
        &[_]f64{ 10_000, 1, 1, 50_003 },
        &[_]f64{ 0.5, -3, 2, -1.5 },
        &[_]f64{ 3, 2, 0.001, 19.001 },
    };
    const solution = try solveSimultaneous(alloc, &equations);
    defer alloc.free(solution);
    try testing.expectEqualSlices(f64, &[_]f64{ 5.0, 2.0, 1.0 }, solution);
}

test "simultaneous linear equation solver: singular systems are rejected" {
    const alloc = testing.allocator;
    const singular = [_][]const f64{
        &[_]f64{ 1, 2, 3 },
        &[_]f64{ 2, 4, 6 },
    };
    try testing.expectError(error.SingularMatrix, solveSimultaneous(alloc, &singular));
}
