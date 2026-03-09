//! Degrees To Radians - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/radians.py

const std = @import("std");
const testing = std.testing;

/// Converts degrees to radians.
/// Time complexity: O(1), Space complexity: O(1)
pub fn radians(degree: f64) f64 {
    return degree / (180.0 / std.math.pi);
}

test "radians: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 3.141592653589793), radians(180), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.6057029118347832), radians(92), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.782202150464463), radians(274), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.9167205845401725), radians(109.82), 1e-12);
}

test "radians: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), radians(0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -std.math.pi / 2.0), radians(-90), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0 * std.math.pi), radians(360), 1e-12);
}
