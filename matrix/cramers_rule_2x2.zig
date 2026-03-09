//! Cramer's Rule 2x2 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/cramers_rule_2x2.py

const std = @import("std");
const testing = std.testing;

pub const CramersRuleError = error{
    InvalidEquation,
    ZeroCoefficients,
    InfiniteSolutions,
    NoSolution,
};

/// Solves a 2x2 linear system represented as `{ a, b, c }` rows.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn cramersRule2x2(equation1: []const f64, equation2: []const f64) CramersRuleError![2]f64 {
    if (equation1.len != 3 or equation2.len != 3) return error.InvalidEquation;
    if (equation1[0] == 0 and equation1[1] == 0 and equation2[0] == 0 and equation2[1] == 0) {
        return error.ZeroCoefficients;
    }

    const a1 = equation1[0];
    const b1 = equation1[1];
    const c1 = equation1[2];
    const a2 = equation2[0];
    const b2 = equation2[1];
    const c2 = equation2[2];

    const determinant = a1 * b2 - a2 * b1;
    const determinant_x = c1 * b2 - c2 * b1;
    const determinant_y = a1 * c2 - a2 * c1;

    if (determinant == 0) {
        if (determinant_x == 0 and determinant_y == 0) return error.InfiniteSolutions;
        return error.NoSolution;
    }
    if (determinant_x == 0 and determinant_y == 0) return .{ 0.0, 0.0 };

    return .{ determinant_x / determinant, determinant_y / determinant };
}

test "cramers rule 2x2: python reference" {
    try testing.expectEqual(@as([2]f64, .{ 0.0, 0.0 }), try cramersRule2x2(&[_]f64{ 2, 3, 0 }, &[_]f64{ 5, 1, 0 }));
    try testing.expectEqual(@as([2]f64, .{ 13.0, 12.5 }), try cramersRule2x2(&[_]f64{ 0, 4, 50 }, &[_]f64{ 2, 0, 26 }));
    try testing.expectEqual(@as([2]f64, .{ 4.0, -7.0 }), try cramersRule2x2(&[_]f64{ 11, 2, 30 }, &[_]f64{ 1, 0, 4 }));
    try testing.expectEqual(@as([2]f64, .{ 2.0, -1.0 }), try cramersRule2x2(&[_]f64{ 4, 7, 1 }, &[_]f64{ 1, 2, 0 }));
}

test "cramers rule 2x2: error paths and extremes" {
    try testing.expectError(error.InfiniteSolutions, cramersRule2x2(&[_]f64{ 1, 2, 3 }, &[_]f64{ 2, 4, 6 }));
    try testing.expectError(error.NoSolution, cramersRule2x2(&[_]f64{ 1, 2, 3 }, &[_]f64{ 2, 4, 7 }));
    try testing.expectError(error.InvalidEquation, cramersRule2x2(&[_]f64{ 1, 2, 3 }, &[_]f64{ 11, 22 }));
    try testing.expectError(error.NoSolution, cramersRule2x2(&[_]f64{ 0, 1, 6 }, &[_]f64{ 0, 0, 3 }));
    try testing.expectError(error.ZeroCoefficients, cramersRule2x2(&[_]f64{ 0, 0, 6 }, &[_]f64{ 0, 0, 3 }));
    try testing.expectError(error.InfiniteSolutions, cramersRule2x2(&[_]f64{ 1, 2, 3 }, &[_]f64{ 1, 2, 3 }));
}
