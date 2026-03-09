//! Searching In Sorted Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/searching_in_sorted_matrix.py

const std = @import("std");
const testing = std.testing;

/// Searches a row/column-sorted matrix from the bottom-left corner.
/// Returns 1-based coordinates on success, or `{-1, -1}` if the key is absent.
///
/// Time complexity: O(r + c)
/// Space complexity: O(1)
pub fn searchInSortedMatrix(comptime T: type, matrix: []const []const T, key: T) [2]isize {
    if (matrix.len == 0 or matrix[0].len == 0) return .{ -1, -1 };
    const cols = matrix[0].len;
    for (matrix) |row| {
        if (row.len != cols) return .{ -1, -1 };
    }

    var row: isize = @intCast(matrix.len - 1);
    var col: usize = 0;
    while (row >= 0 and col < cols) {
        const row_idx: usize = @intCast(row);
        const value = matrix[row_idx][col];
        if (key == value) {
            return .{ row + 1, @as(isize, @intCast(col + 1)) };
        }
        if (key < value) {
            row -= 1;
        } else {
            col += 1;
        }
    }
    return .{ -1, -1 };
}

test "searching in sorted matrix: python reference" {
    const matrix_i = [_][]const i32{
        &[_]i32{ 2, 5, 7 },
        &[_]i32{ 4, 8, 13 },
        &[_]i32{ 9, 11, 15 },
        &[_]i32{ 12, 17, 20 },
    };
    const matrix_f = [_][]const f64{
        &[_]f64{ 2.1, 5, 7 },
        &[_]f64{ 4, 8, 13 },
        &[_]f64{ 9, 11, 15 },
        &[_]f64{ 12, 17, 20 },
    };

    try testing.expectEqual(@as([2]isize, .{ 1, 2 }), searchInSortedMatrix(i32, &matrix_i, 5));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), searchInSortedMatrix(i32, &matrix_i, 21));
    try testing.expectEqual(@as([2]isize, .{ 1, 1 }), searchInSortedMatrix(f64, &matrix_f, 2.1));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), searchInSortedMatrix(f64, &matrix_f, 2.2));
}

test "searching in sorted matrix: boundaries" {
    const single = [_][]const i32{&[_]i32{42}};
    const ragged = [_][]const i32{
        &[_]i32{ 1, 2 },
        &[_]i32{1},
    };

    try testing.expectEqual(@as([2]isize, .{ 1, 1 }), searchInSortedMatrix(i32, &single, 42));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), searchInSortedMatrix(i32, &single, -1));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), searchInSortedMatrix(i32, &[_][]const i32{}, 5));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), searchInSortedMatrix(i32, &ragged, 1));
}
