//! Rotate Matrix 90° Clockwise - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/rotate_matrix.py
//! 90° clockwise = transpose then reverse each row.

const std = @import("std");
const testing = std.testing;

/// Rotates a square n×n matrix 90° clockwise in-place (flat row-major).
pub fn rotate90(mat: []i64, n: usize) void {
    if (n <= 1) return;
    const elem_count = @mulWithOverflow(n, n);
    if (elem_count[1] != 0) return;
    if (mat.len != elem_count[0]) return;

    // Step 1: transpose
    for (0..n) |r| {
        for (r + 1..n) |c| {
            const tmp = mat[r * n + c];
            mat[r * n + c] = mat[c * n + r];
            mat[c * n + r] = tmp;
        }
    }
    // Step 2: reverse each row
    for (0..n) |r| {
        var lo: usize = 0;
        var hi: usize = n - 1;
        while (lo < hi) : ({
            lo += 1;
            hi -= 1;
        }) {
            const tmp = mat[r * n + lo];
            mat[r * n + lo] = mat[r * n + hi];
            mat[r * n + hi] = tmp;
        }
    }
}

test "rotate 90: 4x4" {
    // [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    // → [[13,9,5,1],[14,10,6,2],[15,11,7,3],[16,12,8,4]]
    var mat = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    rotate90(&mat, 4);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 16, 12, 8, 4 }, &mat);
}

test "rotate 90: 3x3" {
    // [[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]]
    var mat = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    rotate90(&mat, 3);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 7, 4, 1, 8, 5, 2, 9, 6, 3 }, &mat);
}

test "rotate 90: 1x1" {
    var mat = [_]i64{5};
    rotate90(&mat, 1);
    try std.testing.expectEqualSlices(i64, &[_]i64{5}, &mat);
}

test "rotate 90: twice is 180" {
    var mat = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    rotate90(&mat, 3);
    rotate90(&mat, 3);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 9, 8, 7, 6, 5, 4, 3, 2, 1 }, &mat);
}

test "rotate 90: n=0 is no-op" {
    var mat = [_]i64{ 1, 2, 3, 4 };
    rotate90(&mat, 0);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4 }, &mat);
}

test "rotate 90: invalid flattened size is no-op" {
    var mat = [_]i64{ 1, 2, 3 };
    rotate90(&mat, 2);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3 }, &mat);
}
