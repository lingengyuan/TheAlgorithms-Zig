//! Static Equilibrium - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/in_static_equilibrium.py

const std = @import("std");
const testing = std.testing;

pub const StaticEquilibriumError = error{
    LengthMismatch,
    NonPositiveEpsilon,
};

pub const Vector2 = struct {
    x: f64,
    y: f64,
};

/// Resolves force magnitude and angle into rectangular components.
/// `radian_mode=false` interprets angle as degrees.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn polarForce(magnitude: f64, angle: f64, radian_mode: bool) Vector2 {
    const theta = if (radian_mode) angle else angle * std.math.pi / 180.0;
    return Vector2{ .x = magnitude * @cos(theta), .y = magnitude * @sin(theta) };
}

/// Checks static equilibrium by summing 2D moments:
/// sum_i (r_i x F_i) ≈ 0.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn inStaticEquilibrium(
    forces: []const Vector2,
    locations: []const Vector2,
    eps: f64,
) StaticEquilibriumError!bool {
    if (forces.len != locations.len) {
        return StaticEquilibriumError.LengthMismatch;
    }
    if (eps <= 0) {
        return StaticEquilibriumError.NonPositiveEpsilon;
    }

    var sum_moments: f64 = 0.0;
    for (forces, locations) |force, location| {
        sum_moments += location.x * force.y - location.y * force.x;
    }

    return @abs(sum_moments) < eps;
}

test "static equilibrium: python examples" {
    const force = [_]Vector2{ .{ .x = 1, .y = 1 }, .{ .x = -1, .y = 2 } };
    const location = [_]Vector2{ .{ .x = 1, .y = 0 }, .{ .x = 10, .y = 0 } };
    try testing.expect(!(try inStaticEquilibrium(&force, &location, 1e-1)));

    const p1 = polarForce(10, 45, false);
    try testing.expectApproxEqAbs(@as(f64, 7.071067811865477), p1.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 7.0710678118654755), p1.y, 1e-12);

    const p2 = polarForce(10, 3.14, true);
    try testing.expectApproxEqAbs(@as(f64, -9.999987317275396), p2.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.01592652916486828), p2.y, 1e-12);
}

test "static equilibrium: additional and extreme cases" {
    const forces1 = [_]Vector2{
        polarForce(718.4, 180 - 30, false),
        polarForce(879.54, 45, false),
        polarForce(100, -90, false),
    };
    const locations1 = [_]Vector2{ .{ .x = 0, .y = 0 }, .{ .x = 0, .y = 0 }, .{ .x = 0, .y = 0 } };
    try testing.expect(try inStaticEquilibrium(&forces1, &locations1, 1e-1));

    const forces2 = [_]Vector2{ .{ .x = 0, .y = -2000 }, .{ .x = 0, .y = -1200 }, .{ .x = 0, .y = 15600 }, .{ .x = 0, .y = -12400 } };
    const locations2 = [_]Vector2{ .{ .x = 0, .y = 0 }, .{ .x = 6, .y = 0 }, .{ .x = 10, .y = 0 }, .{ .x = 12, .y = 0 } };
    try testing.expect(try inStaticEquilibrium(&forces2, &locations2, 1e-1));

    try testing.expectError(StaticEquilibriumError.LengthMismatch, inStaticEquilibrium(&[_]Vector2{.{ .x = 1, .y = 2 }}, &[_]Vector2{}, 1e-1));
    try testing.expectError(StaticEquilibriumError.NonPositiveEpsilon, inStaticEquilibrium(&[_]Vector2{}, &[_]Vector2{}, 0));

    const large_forces = [_]Vector2{ .{ .x = 1e150, .y = 0 }, .{ .x = -1e150, .y = 0 } };
    const large_locations = [_]Vector2{ .{ .x = 1e10, .y = 1e10 }, .{ .x = 1e10, .y = 1e10 } };
    try testing.expect(try inStaticEquilibrium(&large_forces, &large_locations, 1e5));
}
