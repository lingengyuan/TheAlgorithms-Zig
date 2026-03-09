//! Count Negative Numbers In Sorted Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/count_negative_numbers_in_sorted_matrix.py

const std = @import("std");
const testing = std.testing;

/// Returns whether every row and column is sorted in decreasing order.
///
/// Time complexity: O(r * c)
/// Space complexity: O(1)
pub fn validateGrid(grid: []const []const i32) bool {
    if (grid.len == 0) return true;
    const cols = grid[0].len;
    for (grid) |row| {
        if (row.len != cols) return false;
        if (row.len == 0) continue;
        for (1..row.len) |col| {
            if (row[col - 1] < row[col]) return false;
        }
    }

    for (0..cols) |col| {
        for (1..grid.len) |row| {
            if (grid[row - 1][col] < grid[row][col]) return false;
        }
    }
    return true;
}

/// Finds the first index whose value is negative, or `array.len` if none are negative.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn findNegativeIndex(array: []const i32) usize {
    if (array.len == 0 or array[0] < 0) return 0;

    var left: usize = 0;
    var right: usize = array.len - 1;

    while (right + 1 > left) {
        const mid = (left + right) / 2;
        const num = array[mid];

        if (num < 0 and (mid == 0 or array[mid - 1] >= 0)) return mid;
        if (num >= 0) {
            left = mid + 1;
        } else {
            if (mid == 0) return 0;
            right = mid - 1;
        }
    }
    return array.len;
}

/// Counts negatives using the Python module's shrinking-bound binary-search approach.
///
/// Time complexity: O(r log c)
/// Space complexity: O(1)
pub fn countNegativesBinarySearch(grid: []const []const i32) usize {
    if (grid.len == 0 or grid[0].len == 0) return 0;

    var total: usize = 0;
    var bound: usize = grid[0].len;
    for (grid) |row| {
        if (bound > row.len) bound = row.len;
        bound = findNegativeIndex(row[0..bound]);
        total += bound;
    }
    return (grid.len * grid[0].len) - total;
}

/// Brute-force count over every value.
///
/// Time complexity: O(r * c)
/// Space complexity: O(1)
pub fn countNegativesBruteForce(grid: []const []const i32) usize {
    var total: usize = 0;
    for (grid) |row| {
        for (row) |value| {
            if (value < 0) total += 1;
        }
    }
    return total;
}

/// Brute-force count with early row breaks.
///
/// Time complexity: O(r * c)
/// Space complexity: O(1)
pub fn countNegativesBruteForceWithBreak(grid: []const []const i32) usize {
    var total: usize = 0;
    for (grid) |row| {
        for (row, 0..) |value, idx| {
            if (value < 0) {
                total += row.len - idx;
                break;
            }
        }
    }
    return total;
}

test "count negative numbers in sorted matrix: python reference" {
    const grid1 = [_][]const i32{
        &[_]i32{ 4, 3, 2, -1 },
        &[_]i32{ 3, 2, 1, -1 },
        &[_]i32{ 1, 1, -1, -2 },
        &[_]i32{ -1, -1, -2, -3 },
    };
    const grid2 = [_][]const i32{
        &[_]i32{ 3, 2 },
        &[_]i32{ 1, 0 },
    };
    const grid3 = [_][]const i32{&[_]i32{ 7, 7, 6 }};
    const grid4 = [_][]const i32{
        &[_]i32{ 7, 7, 6 },
        &[_]i32{ -1, -2, -3 },
    };

    try testing.expect(validateGrid(&grid1));
    try testing.expectEqual(@as(usize, 8), countNegativesBinarySearch(&grid1));
    try testing.expectEqual(@as(usize, 8), countNegativesBruteForce(&grid1));
    try testing.expectEqual(@as(usize, 8), countNegativesBruteForceWithBreak(&grid1));
    try testing.expectEqual(@as(usize, 0), countNegativesBinarySearch(&grid2));
    try testing.expectEqual(@as(usize, 0), countNegativesBinarySearch(&grid3));
    try testing.expectEqual(@as(usize, 3), countNegativesBinarySearch(&grid4));
}

test "count negative numbers in sorted matrix: helpers and extremes" {
    try testing.expectEqual(@as(usize, 4), findNegativeIndex(&[_]i32{ 0, 0, 0, 0 }));
    try testing.expectEqual(@as(usize, 3), findNegativeIndex(&[_]i32{ 4, 3, 2, -1 }));
    try testing.expectEqual(@as(usize, 2), findNegativeIndex(&[_]i32{ 1, 0, -1, -10 }));
    try testing.expectEqual(@as(usize, 0), findNegativeIndex(&[_]i32{ -1, -1, -2, -3 }));
    try testing.expectEqual(@as(usize, 0), findNegativeIndex(&[_]i32{}));

    const invalid = [_][]const i32{
        &[_]i32{ 3, 2, 1 },
        &[_]i32{ 4, 1, -1 },
    };
    try testing.expect(!validateGrid(&invalid));

    const all_negative = [_][]const i32{
        &[_]i32{ -1, -2 },
        &[_]i32{ -3, -4 },
    };
    try testing.expectEqual(@as(usize, 4), countNegativesBinarySearch(&all_negative));

    const empty = [_][]const i32{};
    try testing.expectEqual(@as(usize, 0), countNegativesBinarySearch(&empty));
}
