//! Project Euler Problem 18: Maximum Path Sum I - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_018/solution.py

const std = @import("std");
const testing = std.testing;

pub const Problem018Error = error{
    EmptyTriangle,
    NonTriangular,
    Overflow,
    OutOfMemory,
};

pub const default_triangle = [_][]const i64{
    &[_]i64{75},
    &[_]i64{ 95, 64 },
    &[_]i64{ 17, 47, 82 },
    &[_]i64{ 18, 35, 87, 10 },
    &[_]i64{ 20, 4, 82, 47, 65 },
    &[_]i64{ 19, 1, 23, 75, 3, 34 },
    &[_]i64{ 88, 2, 77, 73, 7, 63, 67 },
    &[_]i64{ 99, 65, 4, 28, 6, 16, 70, 92 },
    &[_]i64{ 41, 41, 26, 56, 83, 40, 80, 70, 33 },
    &[_]i64{ 41, 48, 72, 33, 47, 32, 37, 16, 94, 29 },
    &[_]i64{ 53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14 },
    &[_]i64{ 70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57 },
    &[_]i64{ 91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48 },
    &[_]i64{ 63, 66, 4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31 },
    &[_]i64{ 4, 62, 98, 27, 23, 9, 70, 98, 73, 93, 38, 53, 60, 4, 23 },
};

fn safeAdd(a: i64, b: i64) Problem018Error!i64 {
    const result = @addWithOverflow(a, b);
    if (result[1] != 0) return Problem018Error.Overflow;
    return result[0];
}

/// Returns the largest top-to-bottom path sum in a triangular grid.
///
/// Time complexity: O(r^2)
/// Space complexity: O(r)
pub fn maxPathSum(triangle: []const []const i64, allocator: std.mem.Allocator) Problem018Error!i64 {
    if (triangle.len == 0) return Problem018Error.EmptyTriangle;

    for (triangle, 0..) |row, i| {
        if (row.len != i + 1) return Problem018Error.NonTriangular;
    }

    var dp = try allocator.alloc(i64, triangle.len);
    defer allocator.free(dp);

    dp[0] = triangle[0][0];

    var i: usize = 1;
    while (i < triangle.len) : (i += 1) {
        const row = triangle[i];

        var j: usize = i;
        while (true) {
            const value = row[j];

            if (j == i) {
                dp[j] = try safeAdd(dp[j - 1], value);
            } else if (j == 0) {
                dp[j] = try safeAdd(dp[j], value);
            } else {
                const parent = if (dp[j - 1] > dp[j]) dp[j - 1] else dp[j];
                dp[j] = try safeAdd(parent, value);
            }

            if (j == 0) break;
            j -= 1;
        }
    }

    var best = dp[0];
    for (1..triangle.len) |idx| {
        if (dp[idx] > best) best = dp[idx];
    }

    return best;
}

/// Euler problem default solution.
pub fn solution(allocator: std.mem.Allocator) Problem018Error!i64 {
    return maxPathSum(&default_triangle, allocator);
}

test "problem 018: python reference" {
    try testing.expectEqual(@as(i64, 1074), try solution(testing.allocator));
}

test "problem 018: statement example and boundaries" {
    const small = [_][]const i64{
        &[_]i64{3},
        &[_]i64{ 7, 4 },
        &[_]i64{ 2, 4, 6 },
        &[_]i64{ 8, 5, 9, 3 },
    };
    try testing.expectEqual(@as(i64, 23), try maxPathSum(&small, testing.allocator));

    const single = [_][]const i64{&[_]i64{42}};
    try testing.expectEqual(@as(i64, 42), try maxPathSum(&single, testing.allocator));

    const empty = [_][]const i64{};
    try testing.expectError(Problem018Error.EmptyTriangle, maxPathSum(&empty, testing.allocator));

    const malformed = [_][]const i64{
        &[_]i64{5},
        &[_]i64{ 1, 2, 3 },
    };
    try testing.expectError(Problem018Error.NonTriangular, maxPathSum(&malformed, testing.allocator));
}

test "problem 018: negative values and overflow-prone input" {
    const negative = [_][]const i64{
        &[_]i64{-1},
        &[_]i64{ -2, -3 },
        &[_]i64{ -4, -5, -6 },
    };
    try testing.expectEqual(@as(i64, -7), try maxPathSum(&negative, testing.allocator));

    const overflow_case = [_][]const i64{
        &[_]i64{std.math.maxInt(i64)},
        &[_]i64{ 1, 1 },
    };
    try testing.expectError(Problem018Error.Overflow, maxPathSum(&overflow_case, testing.allocator));
}
