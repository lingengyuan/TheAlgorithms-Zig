//! Euler Method - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/euler_method.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Calculates explicit Euler approximations for an ODE over a uniform grid.
/// Time complexity: O(steps), Space complexity: O(steps)
pub fn explicitEuler(
    allocator: Allocator,
    ode_func: *const fn (f64, f64) f64,
    y0: f64,
    x0: f64,
    step_size: f64,
    x_end: f64,
) ![]f64 {
    if (!(step_size > 0)) return error.InvalidStepSize;
    if (x_end < x0) return error.InvalidRange;

    const n = @as(usize, @intFromFloat(@ceil((x_end - x0) / step_size)));
    const y = try allocator.alloc(f64, n + 1);
    y[0] = y0;
    var x = x0;

    for (0..n) |k| {
        y[k + 1] = y[k] + step_size * ode_func(x, y[k]);
        x += step_size;
    }

    return y;
}

fn expOde(_: f64, y: f64) f64 {
    return y;
}

test "euler method: python reference example" {
    const alloc = testing.allocator;
    const y = try explicitEuler(alloc, expOde, 1.0, 0.0, 0.01, 5.0);
    defer alloc.free(y);
    try testing.expectApproxEqAbs(@as(f64, 144.77277243257308), y[y.len - 1], 1e-9);
}

test "euler method: edge and extreme cases" {
    const alloc = testing.allocator;
    const single = try explicitEuler(alloc, expOde, 2.5, 1.0, 0.1, 1.0);
    defer alloc.free(single);
    try testing.expectEqual(@as(usize, 1), single.len);
    try testing.expectEqual(@as(f64, 2.5), single[0]);

    try testing.expectError(error.InvalidStepSize, explicitEuler(alloc, expOde, 1.0, 0.0, 0.0, 1.0));
    try testing.expectError(error.InvalidRange, explicitEuler(alloc, expOde, 1.0, 1.0, 0.1, 0.0));

    const fine = try explicitEuler(alloc, expOde, 1.0, 0.0, 0.001, 1.0);
    defer alloc.free(fine);
    try testing.expect(fine[fine.len - 1] > 2.71 and fine[fine.len - 1] < 2.72);
}

