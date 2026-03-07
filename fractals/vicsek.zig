//! Vicsek Fractal Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/fractals/vicsek.py

const std = @import("std");
const testing = std.testing;

pub const Cross = struct {
    x: f64,
    y: f64,
    length: f64,
};

fn drawFractalRecursive(list: *std.ArrayListUnmanaged(Cross), allocator: std.mem.Allocator, x: f64, y: f64, length: f64, depth: usize) !void {
    if (depth == 0) {
        try list.append(allocator, .{ .x = x, .y = y, .length = length });
        return;
    }

    const child_len = length / 3.0;
    try drawFractalRecursive(list, allocator, x, y, child_len, depth - 1);
    try drawFractalRecursive(list, allocator, x + child_len, y, child_len, depth - 1);
    try drawFractalRecursive(list, allocator, x - child_len, y, child_len, depth - 1);
    try drawFractalRecursive(list, allocator, x, y + child_len, child_len, depth - 1);
    try drawFractalRecursive(list, allocator, x, y - child_len, child_len, depth - 1);
}

/// Generates cross centers/size for Vicsek fractal recursion.
/// Caller owns returned slice.
///
/// Time complexity: O(5^depth)
/// Space complexity: O(5^depth)
pub fn generateVicsekFractal(allocator: std.mem.Allocator, x: f64, y: f64, length: f64, depth: usize) ![]Cross {
    var crosses = std.ArrayListUnmanaged(Cross){};
    defer crosses.deinit(allocator);

    try drawFractalRecursive(&crosses, allocator, x, y, length, depth);
    return crosses.toOwnedSlice(allocator);
}

test "vicsek fractal: base and depth-1 layout" {
    const alloc = testing.allocator;

    const d0 = try generateVicsekFractal(alloc, 0, 0, 9, 0);
    defer alloc.free(d0);
    try testing.expectEqual(@as(usize, 1), d0.len);
    try testing.expectApproxEqAbs(@as(f64, 0), d0[0].x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), d0[0].y, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 9), d0[0].length, 1e-12);

    const d1 = try generateVicsekFractal(alloc, 0, 0, 9, 1);
    defer alloc.free(d1);
    try testing.expectEqual(@as(usize, 5), d1.len);

    const expected = [_]Cross{
        .{ .x = 0, .y = 0, .length = 3 },
        .{ .x = 3, .y = 0, .length = 3 },
        .{ .x = -3, .y = 0, .length = 3 },
        .{ .x = 0, .y = 3, .length = 3 },
        .{ .x = 0, .y = -3, .length = 3 },
    };

    for (expected, 0..) |e, i| {
        try testing.expectApproxEqAbs(e.x, d1[i].x, 1e-12);
        try testing.expectApproxEqAbs(e.y, d1[i].y, 1e-12);
        try testing.expectApproxEqAbs(e.length, d1[i].length, 1e-12);
    }
}

test "vicsek fractal: recursive count and extreme depth" {
    const alloc = testing.allocator;

    const d3 = try generateVicsekFractal(alloc, 0, 0, 27, 3);
    defer alloc.free(d3);
    try testing.expectEqual(@as(usize, 125), d3.len);

    const d6 = try generateVicsekFractal(alloc, 0, 0, 729, 6);
    defer alloc.free(d6);
    try testing.expectEqual(@as(usize, 15_625), d6.len);
    try testing.expect(std.math.isFinite(d6[7_812].x));
    try testing.expect(std.math.isFinite(d6[7_812].y));
}
