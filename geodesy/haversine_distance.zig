//! Haversine Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/geodesy/haversine_distance.py

const std = @import("std");
const testing = std.testing;

pub const axis_a = 6_378_137.0;
pub const axis_b = 6_356_752.314245;
pub const radius = 6_378_137.0;

/// Great-circle distance with the same reduced-latitude formulation used in
/// the Python reference.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn haversineDistance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) f64 {
    const flattening = (axis_a - axis_b) / axis_a;
    const phi1 = std.math.atan((1 - flattening) * std.math.tan(std.math.degreesToRadians(lat1)));
    const phi2 = std.math.atan((1 - flattening) * std.math.tan(std.math.degreesToRadians(lat2)));
    const lambda1 = std.math.degreesToRadians(lon1);
    const lambda2 = std.math.degreesToRadians(lon2);

    var sin_sq_phi = std.math.sin((phi2 - phi1) / 2);
    var sin_sq_lambda = std.math.sin((lambda2 - lambda1) / 2);
    sin_sq_phi *= sin_sq_phi;
    sin_sq_lambda *= sin_sq_lambda;

    const h = std.math.sqrt(sin_sq_phi + (std.math.cos(phi1) * std.math.cos(phi2) * sin_sq_lambda));
    return 2 * radius * std.math.asin(h);
}

test "haversine distance: python reference" {
    const distance = haversineDistance(37.774856, -122.424227, 37.864742, -119.537521);
    try testing.expectApproxEqAbs(@as(f64, 254_352.07945444577), distance, 1e-6);
}

test "haversine distance: boundaries" {
    try testing.expectApproxEqAbs(@as(f64, 0), haversineDistance(0, 0, 0, 0), 1e-9);
    try testing.expect(haversineDistance(10, 10, 11, 11) > 0);
}
