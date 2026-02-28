//! Matrix Multiplication - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/matrix_operation.py

const std = @import("std");
const testing = std.testing;

pub const MatrixError = error{DimensionMismatch};

/// Multiplies matrix A (m×k) by matrix B (k×n), returns C (m×n).
/// Matrices are flat row-major slices. Caller owns result.
pub fn matMul(
    allocator: std.mem.Allocator,
    a: []const i64,
    a_rows: usize,
    a_cols: usize,
    b: []const i64,
    b_rows: usize,
    b_cols: usize,
) (MatrixError || std.mem.Allocator.Error)![]i64 {
    if (a_cols != b_rows) return MatrixError.DimensionMismatch;
    _ = a.len;
    _ = b.len;
    const c = try allocator.alloc(i64, a_rows * b_cols);
    @memset(c, 0);
    for (0..a_rows) |i| {
        for (0..b_cols) |j| {
            for (0..a_cols) |k| {
                c[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
        }
    }
    return c;
}

test "matrix multiply: 2x2" {
    const alloc = testing.allocator;
    const a = [_]i64{ 1, 2, 3, 4 };
    const b = [_]i64{ 5, 5, 7, 5 };
    const c = try matMul(alloc, &a, 2, 2, &b, 2, 2);
    defer alloc.free(c);
    try testing.expectEqualSlices(i64, &[_]i64{ 19, 15, 43, 35 }, c);
}

test "matrix multiply: 1x3 × 3x1" {
    const alloc = testing.allocator;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 2, 3, 4 };
    const c = try matMul(alloc, &a, 1, 3, &b, 3, 1);
    defer alloc.free(c);
    try testing.expectEqualSlices(i64, &[_]i64{20}, c);
}

test "matrix multiply: dimension mismatch" {
    const alloc = testing.allocator;
    const a = [_]i64{ 1, 2, 3, 4 };
    const b = [_]i64{ 1, 2, 3 };
    try testing.expectError(MatrixError.DimensionMismatch, matMul(alloc, &a, 2, 2, &b, 3, 1));
}
