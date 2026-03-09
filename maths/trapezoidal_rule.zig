//! Trapezoidal Rule - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/trapezoidal_rule.py

const std = @import("std");
const testing = std.testing;

pub const TrapezoidalRuleError = error{InvalidSteps};

/// Reference polynomial `f(x) = x^2`.
pub fn f(x: f64) f64 {
    return x * x;
}

/// Generates the intermediate points between `a` and `b` with step `h`.
/// Caller owns the returned slice.
pub fn makePoints(allocator: std.mem.Allocator, a: f64, b: f64, h: f64) ![]f64 {
    var points = std.ArrayListUnmanaged(f64){};
    defer points.deinit(allocator);

    var x = a + h;
    while (x <= (b - h)) : (x += h) {
        try points.append(allocator, x);
    }
    return points.toOwnedSlice(allocator);
}

/// Approximates the integral of `f(x)=x^2` on `[boundary[0], boundary[1]]`.
/// Time complexity: O(steps), Space complexity: O(steps)
pub fn trapezoidalRule(
    allocator: std.mem.Allocator,
    boundary: [2]f64,
    steps: usize,
) (TrapezoidalRuleError || std.mem.Allocator.Error)!f64 {
    if (steps == 0) return error.InvalidSteps;
    const h = (boundary[1] - boundary[0]) / @as(f64, @floatFromInt(steps));
    const points = try makePoints(allocator, boundary[0], boundary[1], h);
    defer allocator.free(points);

    var y = (h / 2.0) * f(boundary[0]);
    for (points) |point| {
        y += h * f(point);
    }
    y += (h / 2.0) * f(boundary[1]);
    return y;
}

test "trapezoidal rule: python reference examples" {
    const alloc = testing.allocator;
    try testing.expect(@abs((try trapezoidalRule(alloc, .{ 0, 1 }, 10)) - 0.33333) < 0.01);
    try testing.expect(@abs((try trapezoidalRule(alloc, .{ 0, 1 }, 100)) - 0.33333) < 0.01);
    try testing.expect(@abs((try trapezoidalRule(alloc, .{ 0, 2 }, 1000)) - 2.66667) < 0.01);
}

test "trapezoidal rule: edge and extreme cases" {
    const alloc = testing.allocator;
    const points = try makePoints(alloc, 0, 10, 2.5);
    defer alloc.free(points);
    try testing.expectEqualSlices(f64, &[_]f64{ 2.5, 5.0, 7.5 }, points);
    try testing.expectError(error.InvalidSteps, trapezoidalRule(alloc, .{ 0, 1 }, 0));
}
