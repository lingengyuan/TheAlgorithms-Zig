//! Spiral Print - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/spiral_print.py

const std = @import("std");
const testing = std.testing;

/// Collects matrix elements in clockwise spiral order.
/// `mat` is flat row-major with `rows`×`cols` dimensions. Caller owns result.
pub fn spiralOrder(allocator: std.mem.Allocator, mat: []const i64, rows: usize, cols: usize) ![]i64 {
    const out = try allocator.alloc(i64, rows * cols);
    var top: usize = 0;
    var bottom: usize = rows;
    var left: usize = 0;
    var right: usize = cols;
    var idx: usize = 0;

    while (top < bottom and left < right) {
        // top row left→right
        for (left..right) |c| {
            out[idx] = mat[top * cols + c];
            idx += 1;
        }
        top += 1;
        // right col top→bottom
        for (top..bottom) |r| {
            out[idx] = mat[r * cols + (right - 1)];
            idx += 1;
        }
        right -= 1;
        if (top < bottom) {
            // bottom row right→left
            var c = right;
            while (c > left) {
                c -= 1;
                out[idx] = mat[(bottom - 1) * cols + c];
                idx += 1;
            }
            bottom -= 1;
        }
        if (left < right) {
            // left col bottom→top
            var r = bottom;
            while (r > top) {
                r -= 1;
                out[idx] = mat[r * cols + left];
                idx += 1;
            }
            left += 1;
        }
    }
    return out;
}

test "spiral: 3x4" {
    const alloc = testing.allocator;
    const mat = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const s = try spiralOrder(alloc, &mat, 3, 4);
    defer alloc.free(s);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7 }, s);
}

test "spiral: 3x3" {
    const alloc = testing.allocator;
    const mat = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const s = try spiralOrder(alloc, &mat, 3, 3);
    defer alloc.free(s);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 6, 9, 8, 7, 4, 5 }, s);
}

test "spiral: 1x1" {
    const alloc = testing.allocator;
    const mat = [_]i64{42};
    const s = try spiralOrder(alloc, &mat, 1, 1);
    defer alloc.free(s);
    try testing.expectEqualSlices(i64, &[_]i64{42}, s);
}

test "spiral: 1xN row" {
    const alloc = testing.allocator;
    const mat = [_]i64{ 1, 2, 3, 4 };
    const s = try spiralOrder(alloc, &mat, 1, 4);
    defer alloc.free(s);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4 }, s);
}
