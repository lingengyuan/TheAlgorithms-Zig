//! Chebyshev Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/chebyshev_distance.py

const std = @import("std");
const testing = std.testing;

pub const ChebyshevError = error{ DimensionMismatch, EmptyInput };

/// Returns Chebyshev (chessboard) distance between two points.
/// Time complexity: O(n), Space complexity: O(1)
pub fn chebyshevDistance(point_a: []const f64, point_b: []const f64) ChebyshevError!f64 {
    if (point_a.len != point_b.len) return ChebyshevError.DimensionMismatch;
    if (point_a.len == 0) return ChebyshevError.EmptyInput;

    var max_diff: f64 = 0.0;
    for (point_a, point_b) |a, b| {
        const diff = @abs(a - b);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

test "chebyshev distance: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try chebyshevDistance(&[_]f64{ 1.0, 1.0 }, &[_]f64{ 2.0, 2.0 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 14.2), try chebyshevDistance(&[_]f64{ 1.0, 1.0, 9.0 }, &[_]f64{ 2.0, 2.0, -5.2 }), 1e-12);
    try testing.expectError(ChebyshevError.DimensionMismatch, chebyshevDistance(&[_]f64{1.0}, &[_]f64{ 2.0, 2.0 }));
}

test "chebyshev distance: edge and extreme cases" {
    try testing.expectError(ChebyshevError.EmptyInput, chebyshevDistance(&[_]f64{}, &[_]f64{}));

    var a: [100_000]f64 = undefined;
    var b: [100_000]f64 = undefined;
    @memset(&a, -1.0);
    @memset(&b, 3.5);
    try testing.expectApproxEqAbs(@as(f64, 4.5), try chebyshevDistance(&a, &b), 1e-12);
}
