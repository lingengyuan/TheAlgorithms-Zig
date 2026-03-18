//! Power Iteration - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/power_iteration.py

const std = @import("std");
const testing = std.testing;

pub const PowerIterationError = error{
    NonSquareMatrix,
    DimensionMismatch,
    NonSymmetricMatrix,
    ZeroVector,
    InvalidTolerance,
    InvalidMaxIterations,
    Overflow,
};

pub const PowerIterationResult = struct {
    eigenvalue: f64,
    eigenvector: []f64,

    pub fn deinit(self: PowerIterationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.eigenvector);
    }
};

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

fn vectorNorm(v: []const f64) f64 {
    var sum: f64 = 0;
    for (v) |x| sum += x * x;
    return std.math.sqrt(sum);
}

fn isSymmetric(matrix: []const f64, n: usize) bool {
    for (0..n) |r| {
        for (r + 1..n) |c| {
            if (@abs(matrix[idx(r, c, n)] - matrix[idx(c, r, n)]) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

fn matVecMul(matrix: []const f64, n: usize, vector: []const f64, out: []f64) void {
    for (0..n) |r| {
        var sum: f64 = 0;
        for (0..n) |c| {
            sum += matrix[idx(r, c, n)] * vector[c];
        }
        out[r] = sum;
    }
}

fn rayleigh(matrix: []const f64, n: usize, vector: []const f64) f64 {
    var numerator: f64 = 0;
    var denominator: f64 = 0;

    for (0..n) |i| {
        denominator += vector[i] * vector[i];
        for (0..n) |j| {
            numerator += vector[i] * matrix[idx(i, j, n)] * vector[j];
        }
    }
    return numerator / denominator;
}

/// Computes dominant eigenvalue/eigenvector for real symmetric matrices.
///
/// API note: Python reference also supports Hermitian complex matrices.
/// This Zig implementation currently targets real symmetric input.
///
/// Time complexity: O(max_iterations * n^2)
/// Space complexity: O(n)
pub fn powerIteration(
    allocator: std.mem.Allocator,
    matrix: []const f64,
    rows: usize,
    cols: usize,
    vector: []const f64,
    error_tol: f64,
    max_iterations: usize,
) (PowerIterationError || std.mem.Allocator.Error)!PowerIterationResult {
    if (rows != cols) return PowerIterationError.NonSquareMatrix;

    const count = @mulWithOverflow(rows, cols);
    if (count[1] != 0) return PowerIterationError.Overflow;
    if (matrix.len != count[0]) return PowerIterationError.DimensionMismatch;
    if (vector.len != rows) return PowerIterationError.DimensionMismatch;
    if (error_tol <= 0) return PowerIterationError.InvalidTolerance;
    if (max_iterations == 0) return PowerIterationError.InvalidMaxIterations;
    if (!isSymmetric(matrix, rows)) return PowerIterationError.NonSymmetricMatrix;

    const current = try allocator.dupe(f64, vector);
    errdefer allocator.free(current);

    var norm = vectorNorm(current);
    if (norm <= 1e-12) return PowerIterationError.ZeroVector;
    for (current) |*x| x.* /= norm;

    const w = try allocator.alloc(f64, rows);
    defer allocator.free(w);

    var lambda_prev: f64 = 0;
    var lambda_curr: f64 = 0;

    var it: usize = 0;
    while (it < max_iterations) : (it += 1) {
        matVecMul(matrix, rows, current, w);

        norm = vectorNorm(w);
        if (norm <= 1e-12) {
            return .{ .eigenvalue = 0, .eigenvector = current };
        }

        for (0..rows) |i| {
            current[i] = w[i] / norm;
        }

        lambda_curr = rayleigh(matrix, rows, current);
        const denom = if (@abs(lambda_curr) <= 1e-12) 1.0 else @abs(lambda_curr);
        const rel_error = @abs(lambda_curr - lambda_prev) / denom;

        if (rel_error <= error_tol) {
            return .{ .eigenvalue = lambda_curr, .eigenvector = current };
        }

        lambda_prev = lambda_curr;
    }

    return .{ .eigenvalue = lambda_curr, .eigenvector = current };
}

fn expectApproxSlice(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "power iteration: python real example" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        41, 4,  20,
        4,  26, 30,
        20, 30, 50,
    };
    const vector = [_]f64{ 41, 4, 20 };

    var result = try powerIteration(alloc, &matrix, 3, 3, &vector, 1e-12, 100);
    defer result.deinit(alloc);

    try testing.expectApproxEqAbs(@as(f64, 79.66086378788381), result.eigenvalue, 1e-6);
    try expectApproxSlice(&[_]f64{ 0.44472726, 0.46209842, 0.76725662 }, result.eigenvector, 1e-6);
}

test "power iteration: validation and edge cases" {
    const alloc = testing.allocator;

    const nonsquare = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const v2 = [_]f64{ 1, 2 };
    try testing.expectError(
        PowerIterationError.NonSquareMatrix,
        powerIteration(alloc, &nonsquare, 2, 3, &v2, 1e-12, 100),
    );

    const nonsymmetric = [_]f64{ 1, 2, 3, 4 };
    try testing.expectError(
        PowerIterationError.NonSymmetricMatrix,
        powerIteration(alloc, &nonsymmetric, 2, 2, &v2, 1e-12, 100),
    );

    const symmetric = [_]f64{ 2, 0, 0, 1 };
    try testing.expectError(
        PowerIterationError.ZeroVector,
        powerIteration(alloc, &symmetric, 2, 2, &[_]f64{ 0, 0 }, 1e-12, 100),
    );
    try testing.expectError(
        PowerIterationError.InvalidTolerance,
        powerIteration(alloc, &symmetric, 2, 2, &v2, 0, 100),
    );
    try testing.expectError(
        PowerIterationError.InvalidMaxIterations,
        powerIteration(alloc, &symmetric, 2, 2, &v2, 1e-12, 0),
    );
}
