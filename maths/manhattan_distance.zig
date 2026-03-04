//! Manhattan Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/manhattan_distance.py

const std = @import("std");
const testing = std.testing;

pub const ManhattanError = error{ MissingInput, DimensionMismatch };

/// Returns Manhattan (taxicab) distance between points.
/// Time complexity: O(n), Space complexity: O(1)
pub fn manhattanDistance(point_a: []const f64, point_b: []const f64) ManhattanError!f64 {
    try validatePoint(point_a);
    try validatePoint(point_b);
    if (point_a.len != point_b.len) return ManhattanError.DimensionMismatch;

    var total: f64 = 0.0;
    for (point_a, point_b) |a, b| {
        total += @abs(a - b);
    }
    return total;
}

/// One-liner equivalent variant.
/// Time complexity: O(n), Space complexity: O(1)
pub fn manhattanDistanceOneLiner(point_a: []const f64, point_b: []const f64) ManhattanError!f64 {
    return manhattanDistance(point_a, point_b);
}

fn validatePoint(point: []const f64) ManhattanError!void {
    if (point.len == 0) return ManhattanError.MissingInput;
}

test "manhattan distance: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.0), try manhattanDistance(&[_]f64{ 1, 1 }, &[_]f64{ 2, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try manhattanDistance(&[_]f64{ 1.5, 1.5 }, &[_]f64{ 2, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.5), try manhattanDistance(&[_]f64{ 1.5, 1.5 }, &[_]f64{ 2.5, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 9.0), try manhattanDistance(&[_]f64{ -3, -3, -3 }, &[_]f64{ 0, 0, 0 }), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 2.0), try manhattanDistanceOneLiner(&[_]f64{ 1, 1 }, &[_]f64{ 2, 2 }), 1e-12);
}

test "manhattan distance: edge and extreme cases" {
    try testing.expectError(ManhattanError.MissingInput, manhattanDistance(&[_]f64{ 1, 1 }, &[_]f64{}));
    try testing.expectError(ManhattanError.MissingInput, manhattanDistance(&[_]f64{}, &[_]f64{ 2, 2, 2 }));
    try testing.expectError(ManhattanError.DimensionMismatch, manhattanDistance(&[_]f64{ 1, 1 }, &[_]f64{ 2, 2, 2 }));

    var a: [50_000]f64 = undefined;
    var b: [50_000]f64 = undefined;
    @memset(&a, 0.0);
    @memset(&b, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 50_000.0), try manhattanDistance(&a, &b), 1e-9);
}
