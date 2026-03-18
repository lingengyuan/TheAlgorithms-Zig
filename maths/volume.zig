//! Volume - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/volume.py

const std = @import("std");
const testing = std.testing;

pub const VolumeError = error{
    NegativeValue,
    InvalidRadii,
};

fn ensureNonNegative(values: []const f64) VolumeError!void {
    for (values) |value| {
        if (value < 0) return error.NegativeValue;
    }
}

pub fn volCube(side_length: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{side_length});
    return std.math.pow(f64, side_length, 3);
}

pub fn volSphericalCap(height: f64, radius: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ height, radius });
    return (1.0 / 3.0) * std.math.pi * std.math.pow(f64, height, 2) * (3.0 * radius - height);
}

pub fn volSphere(radius: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{radius});
    return (4.0 / 3.0) * std.math.pi * std.math.pow(f64, radius, 3);
}

pub fn volSpheresIntersect(radius_1: f64, radius_2: f64, centers_distance: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ radius_1, radius_2, centers_distance });
    if (centers_distance == 0) return try volSphere(@min(radius_1, radius_2));

    const h1 = ((radius_1 - radius_2 + centers_distance) * (radius_1 + radius_2 - centers_distance)) / (2.0 * centers_distance);
    const h2 = ((radius_2 - radius_1 + centers_distance) * (radius_2 + radius_1 - centers_distance)) / (2.0 * centers_distance);
    return try volSphericalCap(h1, radius_2) + try volSphericalCap(h2, radius_1);
}

pub fn volSpheresUnion(radius_1: f64, radius_2: f64, centers_distance: f64) VolumeError!f64 {
    if (radius_1 <= 0 or radius_2 <= 0 or centers_distance < 0) return error.InvalidRadii;
    if (centers_distance == 0) return try volSphere(@max(radius_1, radius_2));
    return try volSphere(radius_1) + try volSphere(radius_2) -
        try volSpheresIntersect(radius_1, radius_2, centers_distance);
}

pub fn volCuboid(width: f64, height: f64, length: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ width, height, length });
    return width * height * length;
}

pub fn volCone(area_of_base: f64, height: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ area_of_base, height });
    return area_of_base * height / 3.0;
}

pub fn volRightCircCone(radius: f64, height: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ radius, height });
    return std.math.pi * std.math.pow(f64, radius, 2) * height / 3.0;
}

pub fn volPrism(area_of_base: f64, height: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ area_of_base, height });
    return area_of_base * height;
}

pub fn volPyramid(area_of_base: f64, height: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ area_of_base, height });
    return area_of_base * height / 3.0;
}

pub fn volHemisphere(radius: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{radius});
    return std.math.pow(f64, radius, 3) * std.math.pi * 2.0 / 3.0;
}

pub fn volCircularCylinder(radius: f64, height: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ radius, height });
    return std.math.pow(f64, radius, 2) * height * std.math.pi;
}

pub fn volHollowCircularCylinder(inner_radius: f64, outer_radius: f64, height: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ inner_radius, outer_radius, height });
    if (outer_radius <= inner_radius) return error.InvalidRadii;
    return std.math.pi * (std.math.pow(f64, outer_radius, 2) - std.math.pow(f64, inner_radius, 2)) * height;
}

pub fn volConicalFrustum(height: f64, radius_1: f64, radius_2: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ height, radius_1, radius_2 });
    return (1.0 / 3.0) * std.math.pi * height *
        (std.math.pow(f64, radius_1, 2) + std.math.pow(f64, radius_2, 2) + radius_1 * radius_2);
}

pub fn volTorus(torus_radius: f64, tube_radius: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{ torus_radius, tube_radius });
    return 2.0 * std.math.pi * std.math.pi * torus_radius * std.math.pow(f64, tube_radius, 2);
}

pub fn volIcosahedron(tri_side: f64) VolumeError!f64 {
    try ensureNonNegative(&[_]f64{tri_side});
    return tri_side * tri_side * tri_side * (3.0 + @sqrt(5.0)) * 5.0 / 12.0;
}

test "volume: reference samples" {
    try testing.expectApproxEqAbs(@as(f64, 4.096000000000001), try volCube(1.6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.235987755982988), try volSphericalCap(1, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 21.205750411731103), try volSpheresIntersect(2, 2, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 45.814892864851146), try volSpheresUnion(2, 2, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 14.976), try volCuboid(1.6, 2.6, 3.6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10.0), try volCone(10, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 12.566370614359172), try volRightCircCone(2, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 20.0), try volPrism(10, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10.0), try volPyramid(10, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 523.5987755982989), try volSphere(5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 718.377520120866), try volHemisphere(7), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 150.79644737231007), try volCircularCylinder(4, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 28.274333882308138), try volHollowCircularCylinder(1, 2, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 48_490.482608158454), try volConicalFrustum(45, 7, 28), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 19.739208802178716), try volTorus(1, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 34.088984228514256), try volIcosahedron(2.5), 1e-12);
}

test "volume: invalid and extreme cases" {
    try testing.expectError(error.NegativeValue, volCube(-1));
    try testing.expectError(error.NegativeValue, volSphericalCap(-1, 2));
    try testing.expectError(error.InvalidRadii, volSpheresUnion(0, 2, 1));
    try testing.expectError(error.NegativeValue, volCuboid(1, -2, 3));
    try testing.expectError(error.InvalidRadii, volHollowCircularCylinder(2, 1, 3));
    try testing.expectError(error.NegativeValue, volConicalFrustum(-2, 2, 1));
    try testing.expectError(error.NegativeValue, volIcosahedron(-0.2));

    try testing.expectEqual(@as(f64, 0.0), try volSphere(0));
    try testing.expectEqual(@as(f64, 0.0), try volSpheresIntersect(0, 0, 0));
    try testing.expectEqual(@as(f64, 0.0), try volConicalFrustum(0, 0, 0));
}
