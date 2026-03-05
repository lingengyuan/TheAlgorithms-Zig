//! Schur Complement - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/schur_complement.py

const std = @import("std");
const testing = std.testing;
const matrix_inversion = @import("matrix_inversion.zig");

pub const SchurComplementError = error{
    NonSquareA,
    DimensionMismatch,
    SingularMatrix,
    Overflow,
};

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

fn checkedCount(rows: usize, cols: usize) SchurComplementError!usize {
    const pair = @mulWithOverflow(rows, cols);
    if (pair[1] != 0) return SchurComplementError.Overflow;
    return pair[0];
}

/// Computes Schur complement `S = C - B^T A^{-1} B` for block matrix:
/// [ A  B ]
/// [ B^T C ]
///
/// Accepts flat row-major matrices and an optional pseudo-inverse of `A`.
///
/// Time complexity: O(n^3 + n^2*m + n*m^2), where A is n×n and B is n×m.
/// Space complexity: O(n*m + m^2)
pub fn schurComplement(
    allocator: std.mem.Allocator,
    mat_a: []const f64,
    a_rows: usize,
    a_cols: usize,
    mat_b: []const f64,
    b_rows: usize,
    b_cols: usize,
    mat_c: []const f64,
    c_rows: usize,
    c_cols: usize,
    pseudo_inv: ?[]const f64,
) (SchurComplementError || std.mem.Allocator.Error)![]f64 {
    if (a_rows != a_cols) return SchurComplementError.NonSquareA;
    if (b_rows != a_rows) return SchurComplementError.DimensionMismatch;
    if (c_rows != c_cols) return SchurComplementError.DimensionMismatch;
    if (b_cols != c_rows) return SchurComplementError.DimensionMismatch;

    const a_count = try checkedCount(a_rows, a_cols);
    const b_count = try checkedCount(b_rows, b_cols);
    const c_count = try checkedCount(c_rows, c_cols);
    if (mat_a.len != a_count) return SchurComplementError.DimensionMismatch;
    if (mat_b.len != b_count) return SchurComplementError.DimensionMismatch;
    if (mat_c.len != c_count) return SchurComplementError.DimensionMismatch;

    var owned_inv: ?[]f64 = null;
    defer if (owned_inv) |inv| allocator.free(inv);

    const a_inv: []const f64 = if (pseudo_inv) |provided| blk: {
        if (provided.len != a_count) return SchurComplementError.DimensionMismatch;
        break :blk provided;
    } else blk: {
        const inv = matrix_inversion.invertMatrix(allocator, mat_a, a_rows, a_cols) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.NonSquareMatrix => return SchurComplementError.NonSquareA,
            error.DimensionMismatch => return SchurComplementError.DimensionMismatch,
            error.SingularMatrix => return SchurComplementError.SingularMatrix,
            error.Overflow => return SchurComplementError.Overflow,
        };
        owned_inv = inv;
        break :blk inv;
    };

    const temp_count = try checkedCount(a_rows, b_cols);
    var temp = try allocator.alloc(f64, temp_count);
    defer allocator.free(temp);

    for (0..a_rows) |i| {
        for (0..b_cols) |j| {
            var sum: f64 = 0;
            for (0..a_cols) |k| {
                sum += a_inv[idx(i, k, a_cols)] * mat_b[idx(k, j, b_cols)];
            }
            temp[idx(i, j, b_cols)] = sum;
        }
    }

    const product_count = try checkedCount(c_rows, c_cols);
    var product = try allocator.alloc(f64, product_count);
    defer allocator.free(product);

    for (0..c_rows) |i| {
        for (0..c_cols) |j| {
            var sum: f64 = 0;
            for (0..a_rows) |k| {
                sum += mat_b[idx(k, i, b_cols)] * temp[idx(k, j, b_cols)];
            }
            product[idx(i, j, c_cols)] = sum;
        }
    }

    const out = try allocator.alloc(f64, c_count);
    for (0..c_rows) |r| {
        for (0..c_cols) |c| {
            out[idx(r, c, c_cols)] = mat_c[idx(r, c, c_cols)] - product[idx(r, c, c_cols)];
        }
    }
    return out;
}

fn expectApproxSlice(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "schur complement: python example" {
    const alloc = testing.allocator;

    const a = [_]f64{
        1, 2,
        2, 1,
    };
    const b = [_]f64{
        0, 3,
        3, 0,
    };
    const c = [_]f64{
        2, 1,
        6, 3,
    };

    const s = try schurComplement(alloc, &a, 2, 2, &b, 2, 2, &c, 2, 2, null);
    defer alloc.free(s);

    const expected = [_]f64{
        5, -5,
        0, 6,
    };
    try expectApproxSlice(&expected, s, 1e-9);
}

test "schur complement: dimension validation" {
    const alloc = testing.allocator;

    const a = [_]f64{
        1, 0,
        0, 1,
    };
    const b = [_]f64{
        1, 2,
        3, 4,
    };
    const c_bad = [_]f64{
        1, 2, 3,
        4, 5, 6,
    };
    try testing.expectError(
        SchurComplementError.DimensionMismatch,
        schurComplement(alloc, &a, 2, 2, &b, 2, 2, &c_bad, 2, 3, null),
    );

    try testing.expectError(
        SchurComplementError.DimensionMismatch,
        schurComplement(alloc, &a, 2, 2, &b, 1, 4, &[_]f64{1}, 1, 1, null),
    );
}

test "schur complement: singular A with pseudo inverse and extreme magnitude" {
    const alloc = testing.allocator;

    const singular_a = [_]f64{
        1, 2,
        2, 4,
    };
    const b = [_]f64{
        1,
        1,
    };
    const c = [_]f64{3};

    try testing.expectError(
        SchurComplementError.SingularMatrix,
        schurComplement(alloc, &singular_a, 2, 2, &b, 2, 1, &c, 1, 1, null),
    );

    const pseudo = [_]f64{
        1.0 / 25.0, 2.0 / 25.0,
        2.0 / 25.0, 4.0 / 25.0,
    };
    const s = try schurComplement(alloc, &singular_a, 2, 2, &b, 2, 1, &c, 1, 1, &pseudo);
    defer alloc.free(s);
    try expectApproxSlice(&[_]f64{2.64}, s, 1e-9);

    const a_big = [_]f64{
        1e8, 0,
        0,   2e8,
    };
    const b_big = [_]f64{
        2e3,
        -4e3,
    };
    const c_big = [_]f64{9e2};
    const s_big = try schurComplement(alloc, &a_big, 2, 2, &b_big, 2, 1, &c_big, 1, 1, null);
    defer alloc.free(s_big);
    try testing.expect(s_big[0] < 1_000 and s_big[0] > 700);
}
