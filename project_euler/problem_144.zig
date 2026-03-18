//! Project Euler Problem 144: Investigating Multiple Reflections of a Laser Beam - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_144/sol1.py

const std = @import("std");
const testing = std.testing;

const Reflection = struct {
    x: f64,
    y: f64,
    gradient: f64,
};

fn nextPoint(point_x: f64, point_y: f64, incoming_gradient: f64) Reflection {
    const normal_gradient = point_y / (4.0 * point_x);
    const s2 = 2.0 * normal_gradient / (1.0 + normal_gradient * normal_gradient);
    const c2 = (1.0 - normal_gradient * normal_gradient) / (1.0 + normal_gradient * normal_gradient);
    const outgoing_gradient = (s2 - c2 * incoming_gradient) / (c2 + s2 * incoming_gradient);

    const quadratic_term = outgoing_gradient * outgoing_gradient + 4.0;
    const linear_term = 2.0 * outgoing_gradient * (point_y - outgoing_gradient * point_x);
    const constant_term = std.math.pow(f64, point_y - outgoing_gradient * point_x, 2.0) - 100.0;
    const discriminant = linear_term * linear_term - 4.0 * quadratic_term * constant_term;
    const root = std.math.sqrt(discriminant);

    const x_minus = (-linear_term - root) / (2.0 * quadratic_term);
    const x_plus = (-linear_term + root) / (2.0 * quadratic_term);
    const next_x = if (std.math.approxEqAbs(f64, x_plus, point_x, 1e-12)) x_minus else x_plus;
    const next_y = point_y + outgoing_gradient * (next_x - point_x);

    return .{ .x = next_x, .y = next_y, .gradient = outgoing_gradient };
}

/// Returns the number of reflections before the beam exits the ellipse opening.
/// Time complexity: O(reflections)
/// Space complexity: O(1)
pub fn solution(first_x_coord: f64, first_y_coord: f64) u64 {
    var reflections: u64 = 0;
    var point_x = first_x_coord;
    var point_y = first_y_coord;
    var gradient = (10.1 - point_y) / (0.0 - point_x);

    while (!(point_x >= -0.01 and point_x <= 0.01 and point_y > 0.0)) {
        const next = nextPoint(point_x, point_y, gradient);
        point_x = next.x;
        point_y = next.y;
        gradient = next.gradient;
        reflections += 1;
    }

    return reflections;
}

test "problem 144: python reference" {
    try testing.expectEqual(@as(u64, 1), solution(0.00001, -10.0));
    try testing.expectEqual(@as(u64, 287), solution(5.0, 0.0));
    try testing.expectEqual(@as(u64, 354), solution(1.4, -9.6));
}

test "problem 144: next reflection samples" {
    const reflected_flat = nextPoint(5.0, 0.0, 0.0);
    try testing.expectApproxEqAbs(@as(f64, -5.0), reflected_flat.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), reflected_flat.y, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), reflected_flat.gradient, 1e-12);

    const reflected_sloped = nextPoint(5.0, 0.0, -2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), reflected_sloped.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -10.0), reflected_sloped.y, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), reflected_sloped.gradient, 1e-12);
}
