//! Dodecahedron Formulas - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/dodecahedron.py

const std = @import("std");
const testing = std.testing;

pub const DodecahedronError = error{InvalidEdge};

/// Returns the surface area of a regular dodecahedron.
pub fn dodecahedronSurfaceArea(edge: i64) DodecahedronError!f64 {
    if (edge <= 0) return error.InvalidEdge;
    const edge_f: f64 = @floatFromInt(edge);
    return 3.0 * @sqrt(25.0 + 10.0 * @sqrt(5.0)) * (edge_f * edge_f);
}

/// Returns the volume of a regular dodecahedron.
pub fn dodecahedronVolume(edge: i64) DodecahedronError!f64 {
    if (edge <= 0) return error.InvalidEdge;
    const edge_f: f64 = @floatFromInt(edge);
    return ((15.0 + 7.0 * @sqrt(5.0)) / 4.0) * (edge_f * edge_f * edge_f);
}

test "dodecahedron: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 516.1432201766901), try dodecahedronSurfaceArea(5), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2064.5728807067603), try dodecahedronSurfaceArea(10), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 957.8898700780791), try dodecahedronVolume(5), 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 7663.118960624633), try dodecahedronVolume(10), 1e-10);
}

test "dodecahedron: edge cases" {
    try testing.expectError(error.InvalidEdge, dodecahedronSurfaceArea(-1));
    try testing.expectError(error.InvalidEdge, dodecahedronVolume(0));
}
