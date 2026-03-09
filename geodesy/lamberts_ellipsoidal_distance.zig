//! Lambert's Ellipsoidal Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/geodesy/lamberts_ellipsoidal_distance.py

const std = @import("std");
const testing = std.testing;
const haversine = @import("haversine_distance.zig");

pub const axis_a = 6_378_137.0;
pub const axis_b = 6_356_752.314245;
pub const equatorial_radius = 6_378_137.0;

/// Ellipsoidal distance using Lambert's long-line approximation.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn lambertsEllipsoidalDistance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) !f64 {
    const flattening = (axis_a - axis_b) / axis_a;
    const b_lat1 = std.math.atan((1 - flattening) * std.math.tan(std.math.degreesToRadians(lat1)));
    const b_lat2 = std.math.atan((1 - flattening) * std.math.tan(std.math.degreesToRadians(lat2)));
    const sigma = haversine.haversineDistance(lat1, lon1, lat2, lon2) / equatorial_radius;

    const p = (b_lat1 + b_lat2) / 2;
    const q = (b_lat2 - b_lat1) / 2;

    const x_num = std.math.pow(f64, std.math.sin(p), 2) * std.math.pow(f64, std.math.cos(q), 2);
    const x_den = std.math.pow(f64, std.math.cos(sigma / 2), 2);
    if (std.math.approxEqAbs(f64, x_den, 0.0, 1e-15)) return error.SingularGeometry;
    const x = (sigma - std.math.sin(sigma)) * (x_num / x_den);

    const y_num = std.math.pow(f64, std.math.cos(p), 2) * std.math.pow(f64, std.math.sin(q), 2);
    const y_den = std.math.pow(f64, std.math.sin(sigma / 2), 2);
    if (std.math.approxEqAbs(f64, y_den, 0.0, 1e-15)) return error.SingularGeometry;
    const y = (sigma + std.math.sin(sigma)) * (y_num / y_den);

    return equatorial_radius * (sigma - ((flattening / 2.0) * (x + y)));
}

test "lamberts distance: python reference" {
    try testing.expectApproxEqAbs(@as(f64, 254_351.2128767878), try lambertsEllipsoidalDistance(37.774856, -122.424227, 37.864742, -119.537521), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 4_138_992.016770487), try lambertsEllipsoidalDistance(37.774856, -122.424227, 40.713019, -74.012647), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 9_737_326.376993028), try lambertsEllipsoidalDistance(37.774856, -122.424227, 45.443012, 12.313071), 1e-6);
}

test "lamberts distance: boundaries" {
    try testing.expectError(error.SingularGeometry, lambertsEllipsoidalDistance(0, 0, 0, 0));
}
