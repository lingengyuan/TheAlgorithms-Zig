//! Minkowski Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/minkowski_distance.py

const std = @import("std");
const testing = std.testing;

pub const MinkowskiError = error{ InvalidOrder, DimensionMismatch };

/// Returns Minkowski distance of given `order` between points.
/// Time complexity: O(n), Space complexity: O(1)
pub fn minkowskiDistance(point_a: []const f64, point_b: []const f64, order: i64) MinkowskiError!f64 {
    if (order < 1) return MinkowskiError.InvalidOrder;
    if (point_a.len != point_b.len) return MinkowskiError.DimensionMismatch;

    var sum: f64 = 0.0;
    const order_f: f64 = @floatFromInt(order);
    for (point_a, point_b) |a, b| {
        sum += std.math.pow(f64, @abs(a - b), order_f);
    }

    return std.math.pow(f64, sum, 1.0 / order_f);
}

test "minkowski distance: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 2.0), try minkowskiDistance(&[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 2.0 }, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), try minkowskiDistance(&[_]f64{ 1.0, 2.0, 3.0, 4.0 }, &[_]f64{ 5.0, 6.0, 7.0, 8.0 }, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try minkowskiDistance(&[_]f64{5.0}, &[_]f64{0.0}, 3), 1e-12);
    try testing.expectError(MinkowskiError.InvalidOrder, minkowskiDistance(&[_]f64{1.0}, &[_]f64{2.0}, -1));
    try testing.expectError(MinkowskiError.DimensionMismatch, minkowskiDistance(&[_]f64{1.0}, &[_]f64{ 1.0, 2.0 }, 1));
}

test "minkowski distance: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try minkowskiDistance(&[_]f64{}, &[_]f64{}, 2), 1e-12);

    var a: [10_000]f64 = undefined;
    var b: [10_000]f64 = undefined;
    for (&a, 0..) |*slot, idx| slot.* = @floatFromInt(idx);
    for (&b, 0..) |*slot, idx| slot.* = @floatFromInt(idx + 2);
    try testing.expect((try minkowskiDistance(&a, &b, 3)) > 0.0);
}
