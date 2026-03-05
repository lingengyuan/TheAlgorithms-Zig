//! Conjugate Gradient Method - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/conjugate_gradient.py

const std = @import("std");
const testing = std.testing;

pub const ConjugateGradientError = error{
    InvalidDimension,
    NonSpdMatrix,
    SingularDirection,
    MaxIterationsReached,
    Overflow,
};

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

fn checkedCount(rows: usize, cols: usize) ConjugateGradientError!usize {
    const pair = @mulWithOverflow(rows, cols);
    if (pair[1] != 0) return ConjugateGradientError.Overflow;
    return pair[0];
}

fn isSymmetric(matrix: []const f64, n: usize, tol: f64) bool {
    for (0..n) |r| {
        for (r + 1..n) |c| {
            if (@abs(matrix[idx(r, c, n)] - matrix[idx(c, r, n)]) > tol) return false;
        }
    }
    return true;
}

fn isMatrixSpd(
    allocator: std.mem.Allocator,
    matrix: []const f64,
    n: usize,
    tol: f64,
) std.mem.Allocator.Error!bool {
    if (!isSymmetric(matrix, n, tol)) return false;

    // Cholesky-based positive-definiteness check.
    var l = try allocator.alloc(f64, n * n);
    defer allocator.free(l);
    @memset(l, 0.0);

    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum = matrix[idx(i, j, n)];
            for (0..j) |k| {
                sum -= l[idx(i, k, n)] * l[idx(j, k, n)];
            }

            if (i == j) {
                if (sum <= tol) return false;
                l[idx(i, j, n)] = @sqrt(sum);
            } else {
                const denom = l[idx(j, j, n)];
                if (@abs(denom) <= tol) return false;
                l[idx(i, j, n)] = sum / denom;
            }
        }
    }
    return true;
}

fn dot(a: []const f64, b: []const f64) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

fn matVecMul(matrix: []const f64, n: usize, vec: []const f64, out: []f64) void {
    for (0..n) |r| {
        var sum: f64 = 0;
        for (0..n) |c| {
            sum += matrix[idx(r, c, n)] * vec[c];
        }
        out[r] = sum;
    }
}

/// Solves `A x = b` for SPD matrix `A` using Conjugate Gradient.
///
/// Time complexity: O(iterations * n^2)
/// Space complexity: O(n)
pub fn conjugateGradient(
    allocator: std.mem.Allocator,
    spd_matrix: []const f64,
    n: usize,
    load_vector: []const f64,
    max_iterations: usize,
    tol: f64,
) (ConjugateGradientError || std.mem.Allocator.Error)![]f64 {
    const count = try checkedCount(n, n);
    if (n == 0 or spd_matrix.len != count or load_vector.len != n) {
        return ConjugateGradientError.InvalidDimension;
    }
    if (!try isMatrixSpd(allocator, spd_matrix, n, 1e-9)) {
        return ConjugateGradientError.NonSpdMatrix;
    }

    var x = try allocator.alloc(f64, n);
    errdefer allocator.free(x);
    @memset(x, 0.0);

    var r = try allocator.dupe(f64, load_vector);
    defer allocator.free(r);

    var p = try allocator.dupe(f64, r);
    defer allocator.free(p);

    const w = try allocator.alloc(f64, n);
    defer allocator.free(w);

    var rr = dot(r, r);
    if (@sqrt(rr) <= tol) return x;

    var iter: usize = 0;
    while (iter < max_iterations) : (iter += 1) {
        if (rr <= tol * tol) return x;

        matVecMul(spd_matrix, n, p, w);
        const denom = dot(p, w);
        if (@abs(denom) <= 1e-18) {
            if (rr <= tol * tol) return x;
            return ConjugateGradientError.SingularDirection;
        }

        const alpha = rr / denom;
        var max_dx: f64 = 0;
        for (0..n) |i| {
            const delta = alpha * p[i];
            x[i] += delta;
            max_dx = @max(max_dx, @abs(delta));
        }
        for (0..n) |i| {
            r[i] -= alpha * w[i];
        }

        const rr_new = dot(r, r);
        if (rr_new <= tol * tol) return x;
        const error_residual = @sqrt(rr_new);
        const err_value = @max(error_residual, max_dx);
        if (err_value <= tol) return x;

        const beta = rr_new / rr;
        for (0..n) |i| {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }

    return ConjugateGradientError.MaxIterationsReached;
}

test "conjugate gradient: python example" {
    const alloc = testing.allocator;
    const a = [_]f64{
        8.73256573,  -5.02034289, -2.68709226,
        -5.02034289, 3.78188322,  0.91980451,
        -2.68709226, 0.91980451,  1.94746467,
    };
    const b = [_]f64{ -5.80872761, 3.23807431, 1.95381422 };

    const x = try conjugateGradient(alloc, &a, 3, &b, 1000, 1e-8);
    defer alloc.free(x);
    try testing.expectApproxEqAbs(@as(f64, -0.63114139), x[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -0.01561498), x[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.13979294), x[2], 1e-6);
}

test "conjugate gradient: validation" {
    const alloc = testing.allocator;

    try testing.expectError(
        ConjugateGradientError.InvalidDimension,
        conjugateGradient(alloc, &[_]f64{1}, 2, &[_]f64{ 1, 2 }, 10, 1e-8),
    );

    const non_spd = [_]f64{
        0.34634879, 1.96165514,  2.18277744,
        0.74074469, -1.19648894, -1.34223498,
        -0.7687067, 0.06018373,  -1.16315631,
    };
    try testing.expectError(
        ConjugateGradientError.NonSpdMatrix,
        conjugateGradient(alloc, &non_spd, 3, &[_]f64{ 1, 2, 3 }, 10, 1e-8),
    );
}

test "conjugate gradient: extreme larger SPD system" {
    const alloc = testing.allocator;

    const n: usize = 6;
    const a = [_]f64{
        10, 2, 0, 0, 0, 0,
        2,  9, 1, 0, 0, 0,
        0,  1, 8, 1, 0, 0,
        0,  0, 1, 7, 1, 0,
        0,  0, 0, 1, 6, 1,
        0,  0, 0, 0, 1, 5,
    };
    const x_true = [_]f64{ 1, -2, 3, -1, 2, 4 };
    var b: [n]f64 = undefined;
    for (0..n) |r| {
        var sum: f64 = 0;
        for (0..n) |c| sum += a[idx(r, c, n)] * x_true[c];
        b[r] = sum;
    }

    const x = try conjugateGradient(alloc, &a, n, b[0..], 500, 1e-9);
    defer alloc.free(x);
    for (0..n) |i| try testing.expectApproxEqAbs(x_true[i], x[i], 1e-6);
}
