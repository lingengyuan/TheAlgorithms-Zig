//! LU Decomposition - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/lu_decomposition.py

const std = @import("std");
const testing = std.testing;

pub const LUError = error{ NonSquareMatrix, DimensionMismatch, NoDecomposition, Overflow };

pub const LUResult = struct {
    lower: []f64,
    upper: []f64,

    pub fn deinit(self: LUResult, allocator: std.mem.Allocator) void {
        allocator.free(self.lower);
        allocator.free(self.upper);
    }
};

fn idx(row: usize, col: usize, n: usize) usize {
    return row * n + col;
}

/// Performs LU decomposition (Doolittle form, no pivoting): A = L * U.
///
/// API note: Python reference raises runtime exceptions; this Zig implementation
/// uses explicit error unions.
///
/// Time complexity: O(n^3)
/// Space complexity: O(n^2)
pub fn luDecomposition(
    allocator: std.mem.Allocator,
    table: []const f64,
    rows: usize,
    cols: usize,
) (LUError || std.mem.Allocator.Error)!LUResult {
    if (rows != cols) return LUError.NonSquareMatrix;

    const count = @mulWithOverflow(rows, cols);
    if (count[1] != 0) return LUError.Overflow;
    if (table.len != count[0]) return LUError.DimensionMismatch;

    const n = rows;
    const lower = try allocator.alloc(f64, count[0]);
    errdefer allocator.free(lower);
    const upper = try allocator.alloc(f64, count[0]);
    errdefer allocator.free(upper);

    @memset(lower, 0);
    @memset(upper, 0);

    const epsilon = 1e-12;

    for (0..n) |i| {
        for (0..i) |j| {
            var total: f64 = 0;
            for (0..j) |k| {
                total += lower[idx(i, k, n)] * upper[idx(k, j, n)];
            }

            const pivot = upper[idx(j, j, n)];
            if (@abs(pivot) <= epsilon) return LUError.NoDecomposition;
            lower[idx(i, j, n)] = (table[idx(i, j, n)] - total) / pivot;
        }

        lower[idx(i, i, n)] = 1;

        for (i..n) |j| {
            var total: f64 = 0;
            for (0..i) |k| {
                total += lower[idx(i, k, n)] * upper[idx(k, j, n)];
            }
            upper[idx(i, j, n)] = table[idx(i, j, n)] - total;
        }
    }

    return .{ .lower = lower, .upper = upper };
}

fn expectApproxMatrix(
    expected: []const f64,
    actual: []const f64,
    tolerance: f64,
) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "lu decomposition: 3x3 python example" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        2, -2, 1,
        0, 1,  2,
        5, 3,  1,
    };

    const result = try luDecomposition(alloc, &matrix, 3, 3);
    defer result.deinit(alloc);

    const expected_l = [_]f64{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        2.5, 8.0, 1.0,
    };

    const expected_u = [_]f64{
        2.0, -2.0, 1.0,
        0.0, 1.0,  2.0,
        0.0, 0.0,  -17.5,
    };

    try expectApproxMatrix(&expected_l, result.lower, 1e-9);
    try expectApproxMatrix(&expected_u, result.upper, 1e-9);
}

test "lu decomposition: 2x2 python example" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        4, 3,
        6, 3,
    };

    const result = try luDecomposition(alloc, &matrix, 2, 2);
    defer result.deinit(alloc);

    const expected_l = [_]f64{
        1.0, 0.0,
        1.5, 1.0,
    };

    const expected_u = [_]f64{
        4.0, 3.0,
        0.0, -1.5,
    };

    try expectApproxMatrix(&expected_l, result.lower, 1e-9);
    try expectApproxMatrix(&expected_u, result.upper, 1e-9);
}

test "lu decomposition: non-square and dimension mismatch" {
    const alloc = testing.allocator;

    const non_square = [_]f64{ 2, -2, 1, 0, 1, 2 };
    try testing.expectError(LUError.NonSquareMatrix, luDecomposition(alloc, &non_square, 2, 3));

    const bad_len = [_]f64{ 1, 2, 3 };
    try testing.expectError(LUError.DimensionMismatch, luDecomposition(alloc, &bad_len, 2, 2));
}

test "lu decomposition: no decomposition when leading principal minor is zero" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        0, 1,
        1, 0,
    };

    try testing.expectError(LUError.NoDecomposition, luDecomposition(alloc, &matrix, 2, 2));
}

test "lu decomposition: singular but decomposable case" {
    const alloc = testing.allocator;

    const matrix = [_]f64{
        1, 0,
        1, 0,
    };

    const result = try luDecomposition(alloc, &matrix, 2, 2);
    defer result.deinit(alloc);

    const expected_l = [_]f64{
        1.0, 0.0,
        1.0, 1.0,
    };

    const expected_u = [_]f64{
        1.0, 0.0,
        0.0, 0.0,
    };

    try expectApproxMatrix(&expected_l, result.lower, 1e-9);
    try expectApproxMatrix(&expected_u, result.upper, 1e-9);
}
