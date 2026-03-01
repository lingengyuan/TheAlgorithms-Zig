//! Matrix Exponentiation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/matrix_exponentiation.py

const std = @import("std");
const testing = std.testing;

pub const MatrixError = error{InvalidDimensions};

/// Multiplies two square matrices (n x n), row-major flat representation.
/// Caller owns returned slice.
/// Time complexity: O(n^3), space complexity: O(n^2)
pub fn matrixMultiply(
    allocator: std.mem.Allocator,
    a: []const i64,
    b: []const i64,
    n: usize,
) (MatrixError || std.mem.Allocator.Error)![]i64 {
    if (a.len != n * n or b.len != n * n) return MatrixError.InvalidDimensions;

    const out = try allocator.alloc(i64, n * n);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..n) |i| {
        for (0..n) |k| {
            const aik = a[i * n + k];
            for (0..n) |j| {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }

    return out;
}

fn identityMatrix(allocator: std.mem.Allocator, n: usize) ![]i64 {
    const out = try allocator.alloc(i64, n * n);
    @memset(out, 0);
    for (0..n) |i| out[i * n + i] = 1;
    return out;
}

/// Raises square matrix `matrix` to `exponent` using binary exponentiation.
/// Caller owns returned slice.
/// Time complexity: O(n^3 log exponent), space complexity: O(n^2)
pub fn matrixPower(
    allocator: std.mem.Allocator,
    matrix: []const i64,
    n: usize,
    exponent: u64,
) (MatrixError || std.mem.Allocator.Error)![]i64 {
    if (matrix.len != n * n) return MatrixError.InvalidDimensions;

    var result = try identityMatrix(allocator, n);
    errdefer allocator.free(result);

    if (exponent == 0) return result;

    var base = try allocator.dupe(i64, matrix);
    errdefer allocator.free(base);

    var exp = exponent;
    while (exp > 0) {
        if (exp & 1 == 1) {
            const next = try matrixMultiply(allocator, result, base, n);
            allocator.free(result);
            result = next;
        }

        exp >>= 1;
        if (exp > 0) {
            const squared = try matrixMultiply(allocator, base, base, n);
            allocator.free(base);
            base = squared;
        }
    }

    allocator.free(base);
    return result;
}

test "matrix exponentiation: power zero gives identity" {
    const alloc = testing.allocator;
    const m = [_]i64{ 1, 2, 3, 4 };

    const out = try matrixPower(alloc, &m, 2, 0);
    defer alloc.free(out);

    try testing.expectEqualSlices(i64, &[_]i64{ 1, 0, 0, 1 }, out);
}

test "matrix exponentiation: fibonacci transition matrix" {
    const alloc = testing.allocator;
    const fib_q = [_]i64{ 1, 1, 1, 0 };

    const out = try matrixPower(alloc, &fib_q, 2, 5);
    defer alloc.free(out);

    try testing.expectEqualSlices(i64, &[_]i64{ 8, 5, 5, 3 }, out);
}

test "matrix exponentiation: power one returns original" {
    const alloc = testing.allocator;
    const m = [_]i64{ 2, 3, 5, 7 };

    const out = try matrixPower(alloc, &m, 2, 1);
    defer alloc.free(out);
    try testing.expectEqualSlices(i64, &m, out);
}

test "matrix exponentiation: invalid dimensions" {
    const alloc = testing.allocator;
    try testing.expectError(MatrixError.InvalidDimensions, matrixMultiply(alloc, &[_]i64{ 1, 2, 3 }, &[_]i64{ 4, 5, 6 }, 2));
    try testing.expectError(MatrixError.InvalidDimensions, matrixPower(alloc, &[_]i64{ 1, 2, 3 }, 2, 4));
}

test "matrix exponentiation: extreme exponent on diagonal matrix" {
    const alloc = testing.allocator;
    const diag = [_]i64{ 2, 0, 0, 3 };

    const out = try matrixPower(alloc, &diag, 2, 10);
    defer alloc.free(out);

    try testing.expectEqualSlices(i64, &[_]i64{ 1024, 0, 0, 59049 }, out);
}
