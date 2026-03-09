//! Softmax Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/softmax.py

const std = @import("std");
const testing = std.testing;

/// Applies the softmax function to a vector.
/// Caller owns the returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn softmax(allocator: std.mem.Allocator, vector: []const f64) ![]f64 {
    const out = try allocator.alloc(f64, vector.len);
    if (vector.len == 0) return out;

    var max_value = vector[0];
    for (vector[1..]) |value| {
        if (value > max_value) max_value = value;
    }

    var sum_of_exponents: f64 = 0;
    for (vector, 0..) |value, i| {
        out[i] = @exp(value - max_value);
        sum_of_exponents += out[i];
    }
    for (out) |*value| {
        value.* /= sum_of_exponents;
    }
    return out;
}

test "softmax: python reference examples" {
    const alloc = testing.allocator;
    const out1 = try softmax(alloc, &[_]f64{ 1, 2, 3, 4 });
    defer alloc.free(out1);
    var total: f64 = 0;
    for (out1) |value| total += value;
    try testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-12);

    const out2 = try softmax(alloc, &[_]f64{ 5, 5 });
    defer alloc.free(out2);
    try testing.expectApproxEqAbs(@as(f64, 0.5), out2[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), out2[1], 1e-12);
}

test "softmax: edge and extreme cases" {
    const alloc = testing.allocator;
    const out = try softmax(alloc, &[_]f64{0});
    defer alloc.free(out);
    try testing.expectApproxEqAbs(@as(f64, 1.0), out[0], 1e-12);
    const empty = try softmax(alloc, &[_]f64{});
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}
