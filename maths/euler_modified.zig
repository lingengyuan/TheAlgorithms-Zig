//! Euler Modified - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/euler_modified.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Calculates modified Euler (Heun) approximations for an ODE over a uniform grid.
/// Time complexity: O(steps), Space complexity: O(steps)
pub fn eulerModified(
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
        const y_get = y[k] + step_size * ode_func(x, y[k]);
        y[k + 1] = y[k] + ((step_size / 2.0) * (ode_func(x, y[k]) + ode_func(x + step_size, y_get)));
        x += step_size;
    }

    return y;
}

fn ode1(x: f64, y: f64) f64 {
    return -2.0 * x * (y * y);
}

fn ode2(x: f64, y: f64) f64 {
    return -2.0 * y + (x * x * x) * @exp(-2.0 * x);
}

test "euler modified: python reference examples" {
    const alloc = testing.allocator;
    const y1 = try eulerModified(alloc, ode1, 1.0, 0.0, 0.2, 1.0);
    defer alloc.free(y1);
    try testing.expectApproxEqAbs(@as(f64, 0.503338255442106), y1[y1.len - 1], 1e-12);

    const y2 = try eulerModified(alloc, ode2, 1.0, 0.0, 0.1, 0.3);
    defer alloc.free(y2);
    try testing.expectApproxEqAbs(@as(f64, 0.5525976431951775), y2[y2.len - 1], 1e-12);
}

test "euler modified: edge and extreme cases" {
    const alloc = testing.allocator;
    const single = try eulerModified(alloc, ode1, 3.0, 0.0, 0.5, 0.0);
    defer alloc.free(single);
    try testing.expectEqual(@as(usize, 1), single.len);
    try testing.expectEqual(@as(f64, 3.0), single[0]);

    try testing.expectError(error.InvalidStepSize, eulerModified(alloc, ode1, 1.0, 0.0, -0.1, 1.0));
    try testing.expectError(error.InvalidRange, eulerModified(alloc, ode1, 1.0, 1.0, 0.1, 0.0));

    const stable = try eulerModified(alloc, ode1, 1.0, 0.0, 0.01, 2.0);
    defer alloc.free(stable);
    try testing.expect(stable[stable.len - 1] >= 0.0);
}

