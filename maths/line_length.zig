//! Line Length (Arc Approximation) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/line_length.py

const std = @import("std");
const testing = std.testing;

pub const LineLengthError = error{InvalidSteps};

/// Approximates curve length from `x_start` to `x_end` with linear segments.
/// Time complexity: O(steps), Space complexity: O(1)
pub fn lineLength(
    fnc: *const fn (f64) f64,
    x_start: f64,
    x_end: f64,
    steps: usize,
) LineLengthError!f64 {
    if (steps == 0) return LineLengthError.InvalidSteps;

    var x1 = x_start;
    var fx1 = fnc(x_start);
    var length: f64 = 0.0;

    const step_size = (x_end - x_start) / @as(f64, @floatFromInt(steps));
    var i: usize = 0;
    while (i < steps) : (i += 1) {
        const x2 = x1 + step_size;
        const fx2 = fnc(x2);
        length += std.math.hypot(x2 - x1, fx2 - fx1);
        x1 = x2;
        fx1 = fx2;
    }

    return length;
}

fn identity(x: f64) f64 {
    return x;
}

fn constantOne(_: f64) f64 {
    return 1.0;
}

fn mixedCurve(x: f64) f64 {
    return @sin(5.0 * x) + @cos(10.0 * x) + x * x / 10.0;
}

test "line length: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.414214), try lineLength(identity, 0.0, 1.0, 10), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 10.0), try lineLength(constantOne, -5.5, 4.5, 100), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 69.534930), try lineLength(mixedCurve, 0.0, 10.0, 10_000), 1e-6);
}

test "line length: edge and extreme cases" {
    try testing.expectError(LineLengthError.InvalidSteps, lineLength(identity, 0.0, 1.0, 0));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try lineLength(identity, 5.0, 5.0, 100), 1e-12);
    try testing.expect((try lineLength(mixedCurve, -10.0, 10.0, 200_000)) > 0.0);
}
