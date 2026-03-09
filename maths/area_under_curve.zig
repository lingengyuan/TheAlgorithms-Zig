//! Area Under Curve - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/area_under_curve.py

const std = @import("std");
const testing = std.testing;

pub const AreaUnderCurveError = error{InvalidSteps};

pub const CurveFn = *const fn (f64) f64;

/// Approximates the area under a curve using the trapezoidal rule.
/// Time complexity: O(steps), Space complexity: O(1)
pub fn trapezoidalArea(fnc: CurveFn, x_start: f64, x_end: f64, steps: usize) AreaUnderCurveError!f64 {
    if (steps == 0) return error.InvalidSteps;

    var x1 = x_start;
    var fx1 = fnc(x_start);
    var area: f64 = 0.0;
    var i: usize = 0;
    while (i < steps) : (i += 1) {
        const x2 = (x_end - x_start) / @as(f64, @floatFromInt(steps)) + x1;
        const fx2 = fnc(x2);
        area += @abs(fx2 + fx1) * (x2 - x1) / 2.0;
        x1 = x2;
        fx1 = fx2;
    }
    return area;
}

fn constantFive(_: f64) f64 {
    return 5;
}

fn squareNine(x: f64) f64 {
    return 9 * x * x;
}

test "area under curve: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 10.0), try trapezoidalArea(constantFive, 12.0, 14.0, 1000), 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 192.0), try trapezoidalArea(squareNine, -4.0, 0.0, 10_000), 1e-2);
    try testing.expectApproxEqAbs(@as(f64, 384.0), try trapezoidalArea(squareNine, -4.0, 4.0, 10_000), 1e-2);
}

test "area under curve: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try trapezoidalArea(constantFive, 1.0, 1.0, 10), 1e-12);
    try testing.expectError(error.InvalidSteps, trapezoidalArea(constantFive, 0.0, 1.0, 0));
}
