//! Binary Search Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/binary_search_matrix.py

const std = @import("std");
const testing = std.testing;

/// Recursive binary search over a sorted 1D slice.
///
/// Time complexity: O(log n)
/// Space complexity: O(log n)
pub fn binarySearch(comptime T: type, array: []const T, lower_bound: isize, upper_bound: isize, value: T) isize {
    if (lower_bound > upper_bound or array.len == 0) return -1;

    const mid = lower_bound + @divTrunc(upper_bound - lower_bound, 2);
    const idx: usize = @intCast(mid);

    if (array[idx] == value) return mid;
    if (lower_bound >= upper_bound) return -1;
    if (array[idx] < value) {
        return binarySearch(T, array, mid + 1, upper_bound, value);
    }
    return binarySearch(T, array, lower_bound, mid - 1, value);
}

/// Searches a row/column-sorted matrix by binary-searching each candidate row.
/// Returns `{ row, col }`, or `{-1, -1}` when the value is absent.
///
/// Time complexity: O(r log c)
/// Space complexity: O(log c)
pub fn matBinSearch(comptime T: type, value: T, matrix: []const []const T) [2]isize {
    if (matrix.len == 0) return .{ -1, -1 };

    var row: usize = 0;
    while (row < matrix.len) : (row += 1) {
        const current = matrix[row];
        if (current.len == 0) continue;

        if (current[0] == value) return .{ @intCast(row), 0 };
        if (current[0] > value) break;

        const col = binarySearch(T, current, 0, @as(isize, @intCast(current.len - 1)), value);
        if (col != -1) return .{ @intCast(row), col };
    }

    return .{ -1, -1 };
}

test "binary search matrix: python reference" {
    const matrix = [_][]const i32{
        &[_]i32{ 1, 4, 7, 11, 15 },
        &[_]i32{ 2, 5, 8, 12, 19 },
        &[_]i32{ 3, 6, 9, 16, 22 },
        &[_]i32{ 10, 13, 14, 17, 24 },
        &[_]i32{ 18, 21, 23, 26, 30 },
    };

    try testing.expectEqual(@as([2]isize, .{ 0, 0 }), matBinSearch(i32, 1, &matrix));
    try testing.expectEqual(@as([2]isize, .{ 2, 3 }), matBinSearch(i32, 16, &matrix));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), matBinSearch(i32, 34, &matrix));
}

test "binary search matrix: boundaries" {
    const empty_matrix = [_][]const i32{};
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), matBinSearch(i32, 5, &empty_matrix));

    const ragged = [_][]const i32{
        &[_]i32{},
        &[_]i32{ 10, 12, 14 },
    };
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), matBinSearch(i32, 9, &ragged));
    try testing.expectEqual(@as([2]isize, .{ 1, 1 }), matBinSearch(i32, 12, &ragged));

    const single = [_][]const i32{&[_]i32{42}};
    try testing.expectEqual(@as([2]isize, .{ 0, 0 }), matBinSearch(i32, 42, &single));
    try testing.expectEqual(@as([2]isize, .{ -1, -1 }), matBinSearch(i32, -1, &single));
}
