//! QR Decomposition - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/qr_decomposition.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

pub const QrError = error{
    InvalidMatrix,
    OutOfMemory,
    ZeroNorm,
};

pub const QrResult = struct {
    q: []f64,
    r: []f64,
    rows: usize,
    cols: usize,

    pub fn deinit(self: QrResult, allocator: Allocator) void {
        allocator.free(self.q);
        allocator.free(self.r);
    }
};

/// Computes a Householder QR decomposition for an `m x n` matrix.
/// Matrices are stored in row-major order.
/// Time complexity: O(m * n * min(m, n)), Space complexity: O(m^2 + m*n)
pub fn qrHouseholder(allocator: Allocator, matrix: []const []const f64) QrError!QrResult {
    if (matrix.len == 0 or matrix[0].len == 0) return error.InvalidMatrix;
    const rows = matrix.len;
    const cols = matrix[0].len;
    for (matrix) |row| {
        if (row.len != cols) return error.InvalidMatrix;
    }

    const q = try identityMatrix(allocator, rows);
    errdefer allocator.free(q);
    var r = try allocator.alloc(f64, rows * cols);
    errdefer allocator.free(r);

    for (matrix, 0..) |row, i| {
        for (row, 0..) |value, j| {
            r[i * cols + j] = value;
        }
    }

    const t = @min(rows, cols);
    var k: usize = 0;
    while (k + 1 < t) : (k += 1) {
        var norm_sq: f64 = 0.0;
        var i: usize = k;
        while (i < rows) : (i += 1) {
            const value = r[i * cols + k];
            norm_sq += value * value;
        }
        if (norm_sq <= 1e-18) continue;

        const alpha = @sqrt(norm_sq);
        const sign: f64 = if (r[k * cols + k] >= 0.0) 1.0 else -1.0;
        const v = try allocator.alloc(f64, rows - k);
        defer allocator.free(v);

        v[0] = r[k * cols + k] + sign * alpha;
        i = k + 1;
        while (i < rows) : (i += 1) v[i - k] = r[i * cols + k];

        var v_norm_sq: f64 = 0.0;
        for (v) |value| v_norm_sq += value * value;
        if (v_norm_sq <= 1e-18) return error.ZeroNorm;
        const v_norm = @sqrt(v_norm_sq);
        for (v, 0..) |value, idx| v[idx] = value / v_norm;

        applyHouseholderLeft(r, rows, cols, k, v);
        applyHouseholderRight(q, rows, k, v);
    }

    return .{ .q = q, .r = r, .rows = rows, .cols = cols };
}

fn identityMatrix(allocator: Allocator, size: usize) QrError![]f64 {
    const matrix = try allocator.alloc(f64, size * size);
    @memset(matrix, 0.0);
    for (0..size) |i| matrix[i * size + i] = 1.0;
    return matrix;
}

fn applyHouseholderLeft(r: []f64, rows: usize, cols: usize, k: usize, v: []const f64) void {
    var col: usize = k;
    while (col < cols) : (col += 1) {
        var dot: f64 = 0.0;
        var i: usize = k;
        while (i < rows) : (i += 1) dot += v[i - k] * r[i * cols + col];
        i = k;
        while (i < rows) : (i += 1) {
            r[i * cols + col] -= 2.0 * v[i - k] * dot;
        }
    }
}

fn applyHouseholderRight(q: []f64, rows: usize, k: usize, v: []const f64) void {
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        var dot: f64 = 0.0;
        var col: usize = k;
        while (col < rows) : (col += 1) dot += q[row * rows + col] * v[col - k];
        col = k;
        while (col < rows) : (col += 1) {
            q[row * rows + col] -= 2.0 * dot * v[col - k];
        }
    }
}

fn matrixMultiply(allocator: Allocator, a: []const f64, a_rows: usize, a_cols: usize, b: []const f64, b_cols: usize) ![]f64 {
    const result = try allocator.alloc(f64, a_rows * b_cols);
    @memset(result, 0.0);
    for (0..a_rows) |i| {
        for (0..a_cols) |k| {
            const left = a[i * a_cols + k];
            for (0..b_cols) |j| {
                result[i * b_cols + j] += left * b[k * b_cols + j];
            }
        }
    }
    return result;
}

fn transpose(allocator: Allocator, matrix: []const f64, rows: usize, cols: usize) ![]f64 {
    const result = try allocator.alloc(f64, rows * cols);
    for (0..rows) |i| {
        for (0..cols) |j| result[j * rows + i] = matrix[i * cols + j];
    }
    return result;
}

fn expectMatrixApproxEq(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |left, right| {
        try testing.expectApproxEqAbs(left, right, tolerance);
    }
}

fn expectUpperTriangular(matrix: []const f64, rows: usize, cols: usize, tolerance: f64) !void {
    for (0..rows) |i| {
        for (0..@min(i, cols)) |j| {
            try testing.expect(@abs(matrix[i * cols + j]) <= tolerance);
        }
    }
}

test "qr decomposition: python reference example" {
    const alloc = testing.allocator;
    const input = [_][]const f64{
        &[_]f64{ 12, -51, 4 },
        &[_]f64{ 6, 167, -68 },
        &[_]f64{ -4, 24, -41 },
    };

    var qr = try qrHouseholder(alloc, &input);
    defer qr.deinit(alloc);

    const qr_product = try matrixMultiply(alloc, qr.q, qr.rows, qr.rows, qr.r, qr.cols);
    defer alloc.free(qr_product);
    try expectMatrixApproxEq(&[_]f64{
        12, -51, 4,
        6,  167, -68,
        -4, 24,  -41,
    }, qr_product, 1e-8);

    const q_t = try transpose(alloc, qr.q, qr.rows, qr.rows);
    defer alloc.free(q_t);
    const qtq = try matrixMultiply(alloc, q_t, qr.rows, qr.rows, qr.q, qr.rows);
    defer alloc.free(qtq);
    try expectMatrixApproxEq(&[_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    }, qtq, 1e-8);

    try expectUpperTriangular(qr.r, qr.rows, qr.cols, 1e-8);
}

test "qr decomposition: rectangular matrix" {
    const alloc = testing.allocator;
    const input = [_][]const f64{
        &[_]f64{ 1, 2 },
        &[_]f64{ 3, 4 },
        &[_]f64{ 5, 6 },
    };

    var qr = try qrHouseholder(alloc, &input);
    defer qr.deinit(alloc);
    try testing.expectEqual(@as(usize, 3), qr.rows);
    try testing.expectEqual(@as(usize, 2), qr.cols);

    const qr_product = try matrixMultiply(alloc, qr.q, qr.rows, qr.rows, qr.r, qr.cols);
    defer alloc.free(qr_product);
    try expectMatrixApproxEq(&[_]f64{
        1, 2,
        3, 4,
        5, 6,
    }, qr_product, 1e-8);
}

test "qr decomposition: extreme diagonal scale remains stable" {
    const alloc = testing.allocator;
    const input = [_][]const f64{
        &[_]f64{ 1e9, 1, 0 },
        &[_]f64{ 0, 1e-9, 2 },
        &[_]f64{ 0, 0, -3 },
    };

    var qr = try qrHouseholder(alloc, &input);
    defer qr.deinit(alloc);
    try expectUpperTriangular(qr.r, qr.rows, qr.cols, 1e-7);
}

test "qr decomposition: invalid matrix is rejected" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidMatrix, qrHouseholder(alloc, &[_][]const f64{}));

    const ragged = [_][]const f64{
        &[_]f64{ 1, 2, 3 },
        &[_]f64{ 4, 5 },
    };
    try testing.expectError(error.InvalidMatrix, qrHouseholder(alloc, &ragged));
}
