//! Largest Square Area In Matrix - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/largest_square_area_in_matrix.py

const std = @import("std");
const testing = std.testing;

pub const LargestSquareError = error{
    InvalidGrid,
    OutOfMemory,
};

fn idx(cols: usize, row: usize, col: usize) usize {
    return row * (cols + 1) + col;
}

/// Bottom-up DP for the maximum all-ones square side length.
///
/// Time complexity: O(r * c)
/// Space complexity: O(r * c)
pub fn largestSquareAreaBottomUp(grid: []const []const u8, allocator: std.mem.Allocator) LargestSquareError!usize {
    if (grid.len == 0) return 0;
    const cols = grid[0].len;
    if (cols == 0) return 0;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    const dp = try allocator.alloc(usize, (grid.len + 1) * (cols + 1));
    defer allocator.free(dp);
    @memset(dp, 0);

    var largest: usize = 0;
    var row = grid.len;
    while (row > 0) {
        row -= 1;
        var col = cols;
        while (col > 0) {
            col -= 1;
            if (grid[row][col] == 1) {
                const right = dp[idx(cols, row, col + 1)];
                const diagonal = dp[idx(cols, row + 1, col + 1)];
                const bottom = dp[idx(cols, row + 1, col)];
                const best = 1 + @min(right, @min(diagonal, bottom));
                dp[idx(cols, row, col)] = best;
                largest = @max(largest, best);
            }
        }
    }

    return largest;
}

/// Space-optimized bottom-up DP variant.
///
/// Time complexity: O(r * c)
/// Space complexity: O(c)
pub fn largestSquareAreaBottomUpSpaceOptimized(grid: []const []const u8, allocator: std.mem.Allocator) LargestSquareError!usize {
    if (grid.len == 0) return 0;
    const cols = grid[0].len;
    if (cols == 0) return 0;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    const current = try allocator.alloc(usize, cols + 1);
    defer allocator.free(current);
    const next = try allocator.alloc(usize, cols + 1);
    defer allocator.free(next);
    @memset(current, 0);
    @memset(next, 0);

    var largest: usize = 0;
    var row = grid.len;
    while (row > 0) {
        row -= 1;
        var col = cols;
        while (col > 0) {
            col -= 1;
            if (grid[row][col] == 1) {
                current[col] = 1 + @min(current[col + 1], @min(next[col + 1], next[col]));
                largest = @max(largest, current[col]);
            } else {
                current[col] = 0;
            }
        }
        @memcpy(next, current);
    }

    return largest;
}

test "largest square area in matrix: python reference" {
    const allocator = testing.allocator;

    const full = [_][]const u8{
        &[_]u8{ 1, 1 },
        &[_]u8{ 1, 1 },
    };
    const empty_square = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 0 },
    };

    try testing.expectEqual(@as(usize, 2), try largestSquareAreaBottomUp(&full, allocator));
    try testing.expectEqual(@as(usize, 2), try largestSquareAreaBottomUpSpaceOptimized(&full, allocator));
    try testing.expectEqual(@as(usize, 0), try largestSquareAreaBottomUp(&empty_square, allocator));
    try testing.expectEqual(@as(usize, 0), try largestSquareAreaBottomUpSpaceOptimized(&empty_square, allocator));
}

test "largest square area in matrix: boundaries" {
    const allocator = testing.allocator;

    const mixed = [_][]const u8{
        &[_]u8{ 1, 0, 1, 0, 0 },
        &[_]u8{ 1, 0, 1, 1, 1 },
        &[_]u8{ 1, 1, 1, 1, 1 },
        &[_]u8{ 1, 0, 0, 1, 0 },
    };
    const single = [_][]const u8{&[_]u8{1}};
    const ragged = [_][]const u8{
        &[_]u8{ 1, 1 },
        &[_]u8{1},
    };

    try testing.expectEqual(@as(usize, 2), try largestSquareAreaBottomUp(&mixed, allocator));
    try testing.expectEqual(@as(usize, 2), try largestSquareAreaBottomUpSpaceOptimized(&mixed, allocator));
    try testing.expectEqual(@as(usize, 1), try largestSquareAreaBottomUp(&single, allocator));
    try testing.expectEqual(@as(usize, 0), try largestSquareAreaBottomUp(&[_][]const u8{}, allocator));
    try testing.expectError(error.InvalidGrid, largestSquareAreaBottomUp(&ragged, allocator));
}
