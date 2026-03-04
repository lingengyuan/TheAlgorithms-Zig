//! Minimum Cost Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_cost_path.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MinimumCostPathError = error{
    EmptyMatrix,
    RaggedMatrix,
    Overflow,
};

/// Returns the minimum path sum from top-left to bottom-right
/// moving only right or down.
/// Time complexity: O(rows * cols), Space complexity: O(cols)
pub fn minimumCostPath(
    allocator: Allocator,
    matrix: []const []const i64,
) (MinimumCostPathError || Allocator.Error)!i64 {
    if (matrix.len == 0) return MinimumCostPathError.EmptyMatrix;
    if (matrix[0].len == 0) return MinimumCostPathError.EmptyMatrix;

    const cols = matrix[0].len;
    for (matrix) |row| {
        if (row.len != cols) return MinimumCostPathError.RaggedMatrix;
    }

    const dp = try allocator.alloc(i64, cols);
    defer allocator.free(dp);

    dp[0] = matrix[0][0];
    for (1..cols) |j| {
        const next = @addWithOverflow(dp[j - 1], matrix[0][j]);
        if (next[1] != 0) return MinimumCostPathError.Overflow;
        dp[j] = next[0];
    }

    for (1..matrix.len) |i| {
        const first = @addWithOverflow(dp[0], matrix[i][0]);
        if (first[1] != 0) return MinimumCostPathError.Overflow;
        dp[0] = first[0];

        for (1..cols) |j| {
            const best_prev = @min(dp[j], dp[j - 1]);
            const next = @addWithOverflow(best_prev, matrix[i][j]);
            if (next[1] != 0) return MinimumCostPathError.Overflow;
            dp[j] = next[0];
        }
    }

    return dp[cols - 1];
}

test "minimum cost path: python samples" {
    const matrix1 = [_][]const i64{
        &[_]i64{ 2, 1 },
        &[_]i64{ 3, 1 },
        &[_]i64{ 4, 2 },
    };
    try testing.expectEqual(@as(i64, 6), try minimumCostPath(testing.allocator, &matrix1));

    const matrix2 = [_][]const i64{
        &[_]i64{ 2, 1, 4 },
        &[_]i64{ 2, 1, 3 },
        &[_]i64{ 3, 2, 1 },
    };
    try testing.expectEqual(@as(i64, 7), try minimumCostPath(testing.allocator, &matrix2));
}

test "minimum cost path: boundary matrix" {
    const one = [_][]const i64{
        &[_]i64{42},
    };
    try testing.expectEqual(@as(i64, 42), try minimumCostPath(testing.allocator, &one));
}

test "minimum cost path: invalid shape handling" {
    const ragged = [_][]const i64{
        &[_]i64{ 1, 2, 3 },
        &[_]i64{ 4, 5 },
    };
    try testing.expectError(MinimumCostPathError.RaggedMatrix, minimumCostPath(testing.allocator, &ragged));

    const empty_rows = [_][]const i64{};
    try testing.expectError(MinimumCostPathError.EmptyMatrix, minimumCostPath(testing.allocator, &empty_rows));
}

test "minimum cost path: extreme overflow is detected" {
    const huge = [_][]const i64{
        &[_]i64{ std.math.maxInt(i64), 1 },
        &[_]i64{ 1, 1 },
    };
    try testing.expectError(MinimumCostPathError.Overflow, minimumCostPath(testing.allocator, &huge));
}
