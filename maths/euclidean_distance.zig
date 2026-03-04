//! Euclidean Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/euclidean_distance.py

const std = @import("std");
const testing = std.testing;

pub const EuclideanError = error{DimensionMismatch};

/// Returns Euclidean distance between two vectors.
/// Time complexity: O(n), Space complexity: O(1)
pub fn euclideanDistance(vector_1: []const f64, vector_2: []const f64) EuclideanError!f64 {
    if (vector_1.len != vector_2.len) return EuclideanError.DimensionMismatch;

    var sum: f64 = 0.0;
    for (vector_1, vector_2) |v1, v2| {
        const diff = v1 - v2;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Same computation as euclideanDistance, without external dependencies.
/// Time complexity: O(n), Space complexity: O(1)
pub fn euclideanDistanceNoNp(vector_1: []const f64, vector_2: []const f64) EuclideanError!f64 {
    return euclideanDistance(vector_1, vector_2);
}

test "euclidean distance: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.8284271247461903), try euclideanDistance(&[_]f64{ 0, 0 }, &[_]f64{ 2, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.4641016151377544), try euclideanDistance(&[_]f64{ 0, 0, 0 }, &[_]f64{ 2, 2, 2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), try euclideanDistance(&[_]f64{ 1, 2, 3, 4 }, &[_]f64{ 5, 6, 7, 8 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), try euclideanDistanceNoNp(&[_]f64{ 1, 2, 3, 4 }, &[_]f64{ 5, 6, 7, 8 }), 1e-12);
}

test "euclidean distance: edge and extreme cases" {
    try testing.expectError(EuclideanError.DimensionMismatch, euclideanDistance(&[_]f64{ 1, 2 }, &[_]f64{1}));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try euclideanDistance(&[_]f64{}, &[_]f64{}), 1e-12);

    var a: [10_000]f64 = undefined;
    var b: [10_000]f64 = undefined;
    for (&a, 0..) |*slot, idx| slot.* = @floatFromInt(idx);
    for (&b, 0..) |*slot, idx| slot.* = @floatFromInt(idx + 1);
    try testing.expectApproxEqAbs(@as(f64, 100.0), try euclideanDistance(&a, &b), 1e-9);
}
