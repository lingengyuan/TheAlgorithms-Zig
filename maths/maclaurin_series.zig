//! Maclaurin Series - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/maclaurin_series.py

const std = @import("std");
const testing = std.testing;

pub const MaclaurinError = error{InvalidAccuracy};

fn normalizeAngle(theta: f64) f64 {
    const period = 2.0 * std.math.pi;
    return theta - @floor(theta / period) * period;
}

/// Returns the Maclaurin approximation of sine.
pub fn maclaurinSin(theta: f64, accuracy: i64) MaclaurinError!f64 {
    if (accuracy <= 0) return error.InvalidAccuracy;
    const x = normalizeAngle(theta);
    var term = x;
    var sum = x;
    var n: i64 = 1;
    while (n < accuracy) : (n += 1) {
        const a = @as(f64, @floatFromInt(2 * n));
        const b = @as(f64, @floatFromInt(2 * n + 1));
        term *= -(x * x) / (a * b);
        sum += term;
    }
    return sum;
}

/// Returns the Maclaurin approximation of cosine.
pub fn maclaurinCos(theta: f64, accuracy: i64) MaclaurinError!f64 {
    if (accuracy <= 0) return error.InvalidAccuracy;
    const x = normalizeAngle(theta);
    var term: f64 = 1.0;
    var sum: f64 = 1.0;
    var n: i64 = 1;
    while (n < accuracy) : (n += 1) {
        const a = @as(f64, @floatFromInt(2 * n - 1));
        const b = @as(f64, @floatFromInt(2 * n));
        term *= -(x * x) / (a * b);
        sum += term;
    }
    return sum;
}

test "maclaurin series: python reference examples" {
    try testing.expectApproxEqAbs(std.math.sin(@as(f64, 10.0)), try maclaurinSin(10, 50), 1e-12);
    try testing.expectApproxEqAbs(std.math.sin(@as(f64, -10.0)), try maclaurinSin(-10, 15), 1e-12);
    try testing.expectApproxEqAbs(std.math.cos(@as(f64, 5.0)), try maclaurinCos(5, 50), 1e-12);
    try testing.expectApproxEqAbs(std.math.cos(@as(f64, -10.0)), try maclaurinCos(-10, 15), 1e-12);
}

test "maclaurin series: edge and extreme cases" {
    try testing.expectError(error.InvalidAccuracy, maclaurinSin(10, 0));
    try testing.expectError(error.InvalidAccuracy, maclaurinCos(10, -30));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try maclaurinSin(0, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try maclaurinCos(0, 10), 1e-12);
}
