//! Median Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/median_matrix.py

const std = @import("std");
const testing = std.testing;

pub const MedianMatrixError = error{
    EmptyMatrix,
    OutOfMemory,
};

/// Returns the lower median of all matrix values after flattening and sorting.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n)
pub fn median(allocator: std.mem.Allocator, matrix: []const []const i64) MedianMatrixError!i64 {
    var total_len: usize = 0;
    for (matrix) |row| total_len += row.len;
    if (total_len == 0) return error.EmptyMatrix;

    const linear = try allocator.alloc(i64, total_len);
    defer allocator.free(linear);

    var idx: usize = 0;
    for (matrix) |row| {
        for (row) |value| {
            linear[idx] = value;
            idx += 1;
        }
    }

    std.mem.sort(i64, linear, {}, std.sort.asc(i64));
    return linear[(linear.len - 1) / 2];
}

test "median matrix: python reference" {
    const allocator = testing.allocator;
    const matrix1 = [_][]const i64{
        &[_]i64{ 1, 3, 5 },
        &[_]i64{ 2, 6, 9 },
        &[_]i64{ 3, 6, 9 },
    };
    const matrix2 = [_][]const i64{
        &[_]i64{ 1, 2, 3 },
        &[_]i64{ 4, 5, 6 },
    };

    try testing.expectEqual(@as(i64, 5), try median(allocator, &matrix1));
    try testing.expectEqual(@as(i64, 3), try median(allocator, &matrix2));
}

test "median matrix: boundaries" {
    const allocator = testing.allocator;
    const ragged = [_][]const i64{
        &[_]i64{ 9, 1 },
        &[_]i64{ 3, 7, 5 },
    };

    try testing.expectEqual(@as(i64, 5), try median(allocator, &ragged));
    try testing.expectEqual(@as(i64, -2), try median(allocator, &[_][]const i64{
        &[_]i64{ -5, -2 },
        &[_]i64{ 0, 3 },
    }));
    try testing.expectError(error.EmptyMatrix, median(allocator, &[_][]const i64{}));
}
