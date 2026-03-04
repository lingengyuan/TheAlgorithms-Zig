//! Arc Length - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/arc_length.py

const std = @import("std");
const testing = std.testing;

/// Returns arc length for central angle (degrees) and radius.
/// Time complexity: O(1), Space complexity: O(1)
pub fn arcLength(angle: f64, radius: f64) f64 {
    return 2.0 * std.math.pi * radius * (angle / 360.0);
}

test "arc length: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 3.9269908169872414), arcLength(45, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 31.415926535897928), arcLength(120, 15), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 15.707963267948966), arcLength(90, 10), 1e-12);
}

test "arc length: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), arcLength(0, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), arcLength(90, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -std.math.pi), arcLength(-180, 1), 1e-12);
}
