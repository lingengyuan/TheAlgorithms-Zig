//! Hyperbolic Tangent Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/tanh.py

const std = @import("std");
const testing = std.testing;

/// Applies tanh element-wise to the input vector.
/// Caller owns the returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn tangentHyperbolic(allocator: std.mem.Allocator, vector: []const f64) ![]f64 {
    const out = try allocator.alloc(f64, vector.len);
    for (vector, 0..) |value, i| {
        out[i] = (2.0 / (1.0 + @exp(-2.0 * value))) - 1.0;
    }
    return out;
}

test "tanh: python reference examples" {
    const alloc = testing.allocator;
    const out1 = try tangentHyperbolic(alloc, &[_]f64{ 1, 5, 6, -0.67 });
    defer alloc.free(out1);
    try testing.expectApproxEqAbs(@as(f64, 0.76159416), out1[0], 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 0.9999092), out1[1], 1e-7);
    try testing.expectApproxEqAbs(@as(f64, -0.58497988), out1[3], 1e-8);
}

test "tanh: edge and extreme cases" {
    const alloc = testing.allocator;
    const out = try tangentHyperbolic(alloc, &[_]f64{ 8, 10, 2, -0.98, 13 });
    defer alloc.free(out);
    try testing.expectApproxEqAbs(@as(f64, 1.0), out[1], 1e-8);
    try testing.expectApproxEqAbs(@as(f64, -0.7530659), out[3], 1e-7);
    const empty = try tangentHyperbolic(alloc, &[_]f64{});
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}
