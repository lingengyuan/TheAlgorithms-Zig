//! Index 2D Array In 1D - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/index_2d_array_in_1d.py

const std = @import("std");
const testing = std.testing;

pub const Index2DArrayIterator = struct {
    matrix: []const []const i64,
    row: usize,
    col: usize,

    pub fn init(matrix: []const []const i64) Index2DArrayIterator {
        return .{ .matrix = matrix, .row = 0, .col = 0 };
    }

    /// Returns next flattened element.
    /// Time complexity: amortized O(1), Space complexity: O(1)
    pub fn next(self: *Index2DArrayIterator) ?i64 {
        while (self.row < self.matrix.len) {
            const current_row = self.matrix[self.row];
            if (self.col < current_row.len) {
                const value = current_row[self.col];
                self.col += 1;
                return value;
            }
            self.row += 1;
            self.col = 0;
        }
        return null;
    }
};

/// Retrieves 1D-indexed value from a rectangular 2D array.
/// Time complexity: O(rows), Space complexity: O(1)
pub fn index2DArrayIn1D(array: []const []const i64, index: i64) !i64 {
    const rows = array.len;
    if (rows == 0) return error.NoItemsInArray;

    const cols = array[0].len;
    if (cols == 0) return error.NoItemsInArray;

    for (array[1..]) |row| {
        if (row.len != cols) return error.NonRectangularArray;
    }

    if (index < 0 or index >= @as(i64, @intCast(rows * cols))) return error.IndexOutOfRange;

    const i: usize = @intCast(index);
    return array[i / cols][i % cols];
}

test "index 2d array in 1d: iterator examples" {
    const matrix1 = [_][]const i64{ &[_]i64{5}, &[_]i64{-523}, &[_]i64{-1}, &[_]i64{34}, &[_]i64{0} };
    var it1 = Index2DArrayIterator.init(matrix1[0..]);
    const out1 = [_]i64{ 5, -523, -1, 34, 0 };
    for (out1) |expected| {
        try testing.expectEqual(@as(?i64, expected), it1.next());
    }
    try testing.expectEqual(@as(?i64, null), it1.next());

    const matrix2 = [_][]const i64{ &[_]i64{ 5, 2, 25 }, &[_]i64{ 23, 14, 5 }, &[_]i64{ 324, -1, 0 } };
    var it2 = Index2DArrayIterator.init(matrix2[0..]);
    const out2 = [_]i64{ 5, 2, 25, 23, 14, 5, 324, -1, 0 };
    for (out2) |expected| {
        try testing.expectEqual(@as(?i64, expected), it2.next());
    }
}

test "index 2d array in 1d: function examples and errors" {
    const matrix = [_][]const i64{
        &[_]i64{ 0, 1, 2, 3 },
        &[_]i64{ 4, 5, 6, 7 },
        &[_]i64{ 8, 9, 10, 11 },
    };

    try testing.expectEqual(@as(i64, 5), try index2DArrayIn1D(matrix[0..], 5));
    try testing.expectError(error.IndexOutOfRange, index2DArrayIn1D(matrix[0..], -1));
    try testing.expectError(error.IndexOutOfRange, index2DArrayIn1D(matrix[0..], 12));

    const empty_rows = [_][]const i64{};
    try testing.expectError(error.NoItemsInArray, index2DArrayIn1D(empty_rows[0..], 0));

    const empty_cols = [_][]const i64{&[_]i64{}};
    try testing.expectError(error.NoItemsInArray, index2DArrayIn1D(empty_cols[0..], 0));
}

test "index 2d array in 1d: extreme" {
    const rows: usize = 300;
    const cols: usize = 400;

    var backing = try testing.allocator.alloc(i64, rows * cols);
    defer testing.allocator.free(backing);

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            backing[r * cols + c] = @intCast(r * cols + c);
        }
    }

    var matrix = try testing.allocator.alloc([]const i64, rows);
    defer testing.allocator.free(matrix);
    for (0..rows) |row| {
        matrix[row] = backing[row * cols .. (row + 1) * cols];
    }

    try testing.expectEqual(@as(i64, 0), try index2DArrayIn1D(matrix, 0));
    try testing.expectEqual(@as(i64, 55_555), try index2DArrayIn1D(matrix, 55_555));
    try testing.expectEqual(@as(i64, @intCast(rows * cols - 1)), try index2DArrayIn1D(matrix, @intCast(rows * cols - 1)));
}
