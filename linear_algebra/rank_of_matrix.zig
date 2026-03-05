//! Rank of Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/rank_of_matrix.py

const std = @import("std");
const testing = std.testing;

pub const RankError = error{ DimensionMismatch, Overflow };

fn idx(row: usize, col: usize, cols: usize) usize {
    return row * cols + col;
}

/// Computes matrix rank using Gaussian elimination to row-echelon form.
/// Matrices are flat row-major.
///
/// Time complexity: O(min(r,c) * r * c)
/// Space complexity: O(r * c)
pub fn rankOfMatrix(
    allocator: std.mem.Allocator,
    matrix: []const f64,
    rows: usize,
    cols: usize,
) (RankError || std.mem.Allocator.Error)!usize {
    const count = @mulWithOverflow(rows, cols);
    if (count[1] != 0) return RankError.Overflow;
    if (matrix.len != count[0]) return RankError.DimensionMismatch;

    if (rows == 0 or cols == 0) return 0;

    var a = try allocator.dupe(f64, matrix);
    defer allocator.free(a);

    const eps = 1e-12;
    var rank: usize = 0;
    var row: usize = 0;
    var col: usize = 0;

    while (row < rows and col < cols) {
        var pivot_row = row;
        while (pivot_row < rows and @abs(a[idx(pivot_row, col, cols)]) <= eps) : (pivot_row += 1) {}

        if (pivot_row == rows) {
            col += 1;
            continue;
        }

        if (pivot_row != row) {
            for (0..cols) |c| {
                std.mem.swap(f64, &a[idx(pivot_row, c, cols)], &a[idx(row, c, cols)]);
            }
        }

        const pivot = a[idx(row, col, cols)];
        var r = row + 1;
        while (r < rows) : (r += 1) {
            const factor = a[idx(r, col, cols)] / pivot;
            if (@abs(factor) <= eps) continue;

            var c = col;
            while (c < cols) : (c += 1) {
                a[idx(r, c, cols)] -= factor * a[idx(row, c, cols)];
            }
        }

        rank += 1;
        row += 1;
        col += 1;
    }

    return rank;
}

test "rank of matrix: python examples" {
    const alloc = testing.allocator;

    const m1 = [_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    try testing.expectEqual(@as(usize, 2), try rankOfMatrix(alloc, &m1, 3, 3));

    const m2 = [_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 0,
    };
    try testing.expectEqual(@as(usize, 2), try rankOfMatrix(alloc, &m2, 3, 3));

    const m3 = [_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    try testing.expectEqual(@as(usize, 2), try rankOfMatrix(alloc, &m3, 3, 4));
}

test "rank of matrix: more examples and edge shapes" {
    const alloc = testing.allocator;

    const m4 = [_]f64{
        2, 3,  -1, -1,
        1, -1, -2, 4,
        3, 1,  3,  -2,
        6, 3,  0,  -7,
    };
    try testing.expectEqual(@as(usize, 4), try rankOfMatrix(alloc, &m4, 4, 4));

    const m5 = [_]f64{
        2, 1,  -3, -6,
        3, -3, 1,  2,
        1, 1,  1,  2,
    };
    try testing.expectEqual(@as(usize, 3), try rankOfMatrix(alloc, &m5, 3, 4));

    const m6 = [_]f64{
        2, -1, 0,
        1, 3,  4,
        4, 1,  -3,
    };
    try testing.expectEqual(@as(usize, 3), try rankOfMatrix(alloc, &m6, 3, 3));

    const m7 = [_]f64{
        3,  2,  1,
        -6, -4, -2,
    };
    try testing.expectEqual(@as(usize, 1), try rankOfMatrix(alloc, &m7, 2, 3));

    try testing.expectEqual(@as(usize, 0), try rankOfMatrix(alloc, &[_]f64{}, 2, 0));
    try testing.expectEqual(@as(usize, 1), try rankOfMatrix(alloc, &[_]f64{1}, 1, 1));
    try testing.expectEqual(@as(usize, 0), try rankOfMatrix(alloc, &[_]f64{}, 1, 0));
}

test "rank of matrix: extreme all-zero and dimension mismatch" {
    const alloc = testing.allocator;

    var zeros: [25]f64 = undefined;
    @memset(zeros[0..], 0.0);
    try testing.expectEqual(@as(usize, 0), try rankOfMatrix(alloc, zeros[0..], 5, 5));

    try testing.expectError(RankError.DimensionMismatch, rankOfMatrix(alloc, &[_]f64{ 1, 2, 3 }, 2, 2));
}
