//! Matrix Transpose - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/matrix_operation.py

const std = @import("std");
const testing = std.testing;

/// Transposes an m×n matrix (flat row-major) into n×m. Caller owns result.
pub fn transpose(
    allocator: std.mem.Allocator,
    mat: []const i64,
    rows: usize,
    cols: usize,
) ![]i64 {
    const out = try allocator.alloc(i64, rows * cols);
    for (0..rows) |r| {
        for (0..cols) |c| {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    return out;
}

test "transpose: 2x3 → 3x2" {
    const alloc = testing.allocator;
    // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    const mat = [_]i64{ 1, 2, 3, 4, 5, 6 };
    const t = try transpose(alloc, &mat, 2, 3);
    defer alloc.free(t);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 4, 2, 5, 3, 6 }, t);
}

test "transpose: 1x1" {
    const alloc = testing.allocator;
    const mat = [_]i64{42};
    const t = try transpose(alloc, &mat, 1, 1);
    defer alloc.free(t);
    try testing.expectEqualSlices(i64, &[_]i64{42}, t);
}

test "transpose: square 3x3" {
    const alloc = testing.allocator;
    // [[1,2,3],[4,5,6],[7,8,9]] → [[1,4,7],[2,5,8],[3,6,9]]
    const mat = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const t = try transpose(alloc, &mat, 3, 3);
    defer alloc.free(t);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 4, 7, 2, 5, 8, 3, 6, 9 }, t);
}
