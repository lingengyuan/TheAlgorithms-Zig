//! Polynomial Evaluation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/polynomial_evaluation.py

const std = @import("std");
const testing = std.testing;

/// Evaluates polynomial at x using direct power summation.
/// Coefficients are in ascending degree order.
/// Time complexity: O(n log i), Space complexity: O(1)
pub fn evaluatePoly(poly: []const f64, x: f64) f64 {
    var total: f64 = 0.0;
    for (poly, 0..) |coeff, i| {
        total += coeff * std.math.pow(f64, x, @floatFromInt(i));
    }
    return total;
}

/// Evaluates polynomial at x using Horner's method.
/// Time complexity: O(n), Space complexity: O(1)
pub fn horner(poly: []const f64, x: f64) f64 {
    var result: f64 = 0.0;
    var i: usize = poly.len;
    while (i > 0) {
        i -= 1;
        result = result * x + poly[i];
    }
    return result;
}

test "polynomial evaluation: python reference examples" {
    const poly = [_]f64{ 0.0, 0.0, 5.0, 9.3, 7.0 };
    try testing.expectApproxEqAbs(@as(f64, 79_800.0), evaluatePoly(&poly, 10.0), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 79_800.0), horner(&poly, 10.0), 1e-9);
}

test "polynomial evaluation: edge and extreme cases" {
    const poly = [_]f64{ 0.0, 0.0, 5.0, 9.3, 7.0 };
    try testing.expectApproxEqAbs(@as(f64, 180_339.9), evaluatePoly(&poly, -13.0), 1e-6);
    try testing.expectApproxEqAbs(evaluatePoly(&poly, -13.0), horner(&poly, -13.0), 1e-6);

    try testing.expectApproxEqAbs(@as(f64, 0.0), evaluatePoly(&[_]f64{}, 10.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), horner(&[_]f64{}, 10.0), 1e-12);
}
