//! 2D Transformations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/transformations_2d.py

const std = @import("std");
const testing = std.testing;

pub const Matrix2 = [2][2]f64;

/// Returns 2D scaling matrix.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn scaling(scaling_factor: f64) Matrix2 {
    return .{
        .{ scaling_factor, 0.0 },
        .{ 0.0, scaling_factor },
    };
}

/// Returns 2D rotation matrix.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn rotation(angle: f64) Matrix2 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .{ c, -s },
        .{ s, c },
    };
}

/// Returns 2D projection matrix.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn projection(angle: f64) Matrix2 {
    const c = @cos(angle);
    const s = @sin(angle);
    const cs = c * s;
    return .{
        .{ c * c, cs },
        .{ cs, s * s },
    };
}

/// Returns 2D reflection matrix.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn reflection(angle: f64) Matrix2 {
    const c = @cos(angle);
    const s = @sin(angle);
    const cs = c * s;
    return .{
        .{ 2.0 * c * c - 1.0, 2.0 * cs },
        .{ 2.0 * cs, 2.0 * s * s - 1.0 },
    };
}

fn expectApproxMatrix(expected: Matrix2, got: Matrix2, tolerance: f64) !void {
    for (0..2) |r| {
        for (0..2) |c| {
            try testing.expectApproxEqAbs(expected[r][c], got[r][c], tolerance);
        }
    }
}

test "transformations 2d: python examples" {
    const expected_scaling = Matrix2{
        .{ 5.0, 0.0 },
        .{ 0.0, 5.0 },
    };
    try expectApproxMatrix(expected_scaling, scaling(5), 1e-12);

    const angle: f64 = 45.0;
    const expected_rotation = Matrix2{
        .{ 0.5253219888177297, -0.8509035245341184 },
        .{ 0.8509035245341184, 0.5253219888177297 },
    };
    try expectApproxMatrix(expected_rotation, rotation(angle), 1e-12);

    const expected_projection = Matrix2{
        .{ 0.27596319193541496, 0.446998331800279 },
        .{ 0.446998331800279, 0.7240368080645851 },
    };
    try expectApproxMatrix(expected_projection, projection(angle), 1e-12);

    const expected_reflection = Matrix2{
        .{ -0.4480736161291701, 0.893996663600558 },
        .{ 0.893996663600558, 0.4480736161291701 },
    };
    try expectApproxMatrix(expected_reflection, reflection(angle), 1e-12);
}

test "transformations 2d: boundary angles" {
    const expected_rotation_zero = Matrix2{
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
    };
    try expectApproxMatrix(expected_rotation_zero, rotation(0.0), 1e-12);

    const expected_projection_zero = Matrix2{
        .{ 1.0, 0.0 },
        .{ 0.0, 0.0 },
    };
    try expectApproxMatrix(expected_projection_zero, projection(0.0), 1e-12);

    const expected_reflection_zero = Matrix2{
        .{ 1.0, 0.0 },
        .{ 0.0, -1.0 },
    };
    try expectApproxMatrix(expected_reflection_zero, reflection(0.0), 1e-12);
}

test "transformations 2d: extreme scaling magnitude" {
    const m = scaling(1e12);
    try testing.expectApproxEqAbs(@as(f64, 1e12), m[0][0], 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 1e12), m[1][1], 1e-3);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[0][1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[1][0], 1e-12);
}

test "transformations 2d: reflection is involutive" {
    const r = reflection(0.731);
    const rr = Matrix2{
        .{
            r[0][0] * r[0][0] + r[0][1] * r[1][0],
            r[0][0] * r[0][1] + r[0][1] * r[1][1],
        },
        .{
            r[1][0] * r[0][0] + r[1][1] * r[1][0],
            r[1][0] * r[0][1] + r[1][1] * r[1][1],
        },
    };
    try expectApproxMatrix(Matrix2{ .{ 1.0, 0.0 }, .{ 0.0, 1.0 } }, rr, 1e-12);
}
