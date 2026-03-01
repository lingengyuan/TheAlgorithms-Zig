//! Matrix Multiplication - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/matrix_operation.py

const std = @import("std");
const testing = std.testing;

pub const MatrixError = error{ DimensionMismatch, InvalidMatrixSize, Overflow };

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
    const a_count = @mulWithOverflow(a_rows, a_cols);
    if (a_count[1] != 0) return MatrixError.Overflow;
    const b_count = @mulWithOverflow(b_rows, b_cols);
    if (b_count[1] != 0) return MatrixError.Overflow;
    const c_count = @mulWithOverflow(a_rows, b_cols);
    if (c_count[1] != 0) return MatrixError.Overflow;
    if (a.len != a_count[0] or b.len != b_count[0]) return MatrixError.InvalidMatrixSize;

    const c = try allocator.alloc(i64, c_count[0]);
    errdefer allocator.free(c);
    @memset(c, 0);
    for (0..a_rows) |i| {
        for (0..b_cols) |j| {
            for (0..a_cols) |k| {
                const mul = @mulWithOverflow(a[i * a_cols + k], b[k * b_cols + j]);
                if (mul[1] != 0) return MatrixError.Overflow;
                const sum = @addWithOverflow(c[i * b_cols + j], mul[0]);
                if (sum[1] != 0) return MatrixError.Overflow;
                c[i * b_cols + j] = sum[0];
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

test "matrix multiply: invalid flattened size" {
    const alloc = testing.allocator;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 1, 2, 3, 4 };
    try testing.expectError(MatrixError.InvalidMatrixSize, matMul(alloc, &a, 2, 2, &b, 2, 2));
}

test "matrix multiply: overflow is reported" {
    const alloc = testing.allocator;
    const a = [_]i64{std.math.maxInt(i64)};
    const b = [_]i64{2};
    try testing.expectError(MatrixError.Overflow, matMul(alloc, &a, 1, 1, &b, 1, 1));
}
