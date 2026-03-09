//! Sigmoid Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sigmoid.py

const std = @import("std");
const testing = std.testing;

/// Applies the sigmoid function element-wise.
/// Caller owns the returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn sigmoid(allocator: std.mem.Allocator, vector: []const f64) ![]f64 {
    const out = try allocator.alloc(f64, vector.len);
    for (vector, 0..) |value, i| {
        out[i] = 1.0 / (1.0 + @exp(-value));
    }
    return out;
}

test "sigmoid: python reference examples" {
    const alloc = testing.allocator;
    const out1 = try sigmoid(alloc, &[_]f64{ -1.0, 1.0, 2.0 });
    defer alloc.free(out1);
    try testing.expectApproxEqAbs(@as(f64, 0.2689414213699951), out1[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), out1[1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.8807970779778823), out1[2], 1e-12);
}

test "sigmoid: edge and extreme cases" {
    const alloc = testing.allocator;
    const out = try sigmoid(alloc, &[_]f64{0.0});
    defer alloc.free(out);
    try testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    const empty = try sigmoid(alloc, &[_]f64{});
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}
