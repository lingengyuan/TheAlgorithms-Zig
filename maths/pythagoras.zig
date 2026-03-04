//! Pythagoras Distance in 3D - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/pythagoras.py

const std = @import("std");
const testing = std.testing;

pub const Point = struct {
    x: f64,
    y: f64,
    z: f64,
};

/// Returns Euclidean distance between two 3D points.
/// Time complexity: O(1), Space complexity: O(1)
pub fn distance(a: Point, b: Point) f64 {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dz = b.z - a.z;
    return @sqrt(@abs(dx * dx + dy * dy + dz * dz));
}

test "pythagoras: python reference example" {
    const point1 = Point{ .x = 2, .y = -1, .z = 7 };
    const point2 = Point{ .x = 1, .y = -3, .z = 5 };
    try testing.expectApproxEqAbs(@as(f64, 3.0), distance(point1, point2), 1e-12);
}

test "pythagoras: edge and extreme cases" {
    const p = Point{ .x = 0, .y = 0, .z = 0 };
    try testing.expectApproxEqAbs(@as(f64, 0.0), distance(p, p), 1e-12);

    const a = Point{ .x = 1e9, .y = -1e9, .z = 1e9 };
    const b = Point{ .x = -1e9, .y = 1e9, .z = -1e9 };
    try testing.expect(distance(a, b) > 0.0);
}
