//! Points Are Collinear In 3D - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/points_are_collinear_3d.py

const std = @import("std");
const testing = std.testing;

pub const Vector3d = struct {
    x: f64,
    y: f64,
    z: f64,
};

pub const Point3d = Vector3d;

/// Creates the vector from `end_point1` to `end_point2`.
pub fn createVector(end_point1: Point3d, end_point2: Point3d) Vector3d {
    return .{
        .x = end_point2.x - end_point1.x,
        .y = end_point2.y - end_point1.y,
        .z = end_point2.z - end_point1.z,
    };
}

/// Returns the 3D cross-product of two vectors.
pub fn get3dVectorsCross(ab: Vector3d, ac: Vector3d) Vector3d {
    return .{
        .x = ab.y * ac.z - ab.z * ac.y,
        .y = -((ab.x * ac.z) - (ab.z * ac.x)),
        .z = ab.x * ac.y - ab.y * ac.x,
    };
}

/// Returns true when the vector rounds to `(0, 0, 0)` at the chosen accuracy.
pub fn isZeroVector(vector: Vector3d, accuracy: u32) bool {
    const scale = std.math.pow(f64, 10.0, @floatFromInt(accuracy));
    return roundTo(vector.x, scale) == 0.0 and roundTo(vector.y, scale) == 0.0 and roundTo(vector.z, scale) == 0.0;
}

fn roundTo(value: f64, scale: f64) f64 {
    return @round(value * scale) / scale;
}

/// Returns true when the three points are collinear.
pub fn areCollinear(a: Point3d, b: Point3d, c: Point3d, accuracy: u32) bool {
    const ab = createVector(a, b);
    const ac = createVector(a, c);
    return isZeroVector(get3dVectorsCross(ab, ac), accuracy);
}

test "points collinear 3d: python reference examples" {
    try testing.expect(areCollinear(
        .{ .x = 4.802293498137402, .y = 3.536233125455244, .z = 0 },
        .{ .x = -2.186788107953106, .y = -9.24561398001649, .z = 7.141509524846482 },
        .{ .x = 1.530169574640268, .y = -2.447927606600034, .z = 3.343487096469054 },
        10,
    ));
    try testing.expect(!areCollinear(
        .{ .x = 2.399001826862445, .y = -2.452009976680793, .z = 4.464656666157666 },
        .{ .x = -3.682816335934376, .y = 5.753788986533145, .z = 9.490993909044244 },
        .{ .x = 1.962903518985307, .y = 3.741415730125627, .z = 7 },
        10,
    ));
}

test "points collinear 3d: edge cases" {
    const cross = get3dVectorsCross(.{ .x = 1, .y = 1, .z = 1 }, .{ .x = 1, .y = 1, .z = 1 });
    try testing.expect(isZeroVector(cross, 10));
    try testing.expect(areCollinear(.{ .x = 0, .y = 0, .z = 0 }, .{ .x = 1, .y = 1, .z = 1 }, .{ .x = 2, .y = 2, .z = 2 }, 10));
}
