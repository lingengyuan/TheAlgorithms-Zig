//! Sierpinski Triangle Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/fractals/sierpinski_triangle.py

const std = @import("std");
const testing = std.testing;

pub const Point = struct {
    x: f64,
    y: f64,
};

pub const Triangle = struct {
    v1: Point,
    v2: Point,
    v3: Point,
};

/// Returns midpoint between two points.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn getMid(p1: Point, p2: Point) Point {
    return .{
        .x = (p1.x + p2.x) / 2.0,
        .y = (p1.y + p2.y) / 2.0,
    };
}

fn buildTriangles(
    list: *std.ArrayListUnmanaged(Triangle),
    allocator: std.mem.Allocator,
    vertex1: Point,
    vertex2: Point,
    vertex3: Point,
    depth: usize,
) !void {
    try list.append(allocator, .{
        .v1 = vertex1,
        .v2 = vertex2,
        .v3 = vertex3,
    });

    if (depth == 0) {
        return;
    }

    try buildTriangles(list, allocator, vertex1, getMid(vertex1, vertex2), getMid(vertex1, vertex3), depth - 1);
    try buildTriangles(list, allocator, vertex2, getMid(vertex1, vertex2), getMid(vertex2, vertex3), depth - 1);
    try buildTriangles(list, allocator, vertex3, getMid(vertex3, vertex2), getMid(vertex1, vertex3), depth - 1);
}

/// Generates recursive triangle geometry in the same traversal order as the
/// Python reference function.
/// Caller owns returned slice.
///
/// Time complexity: O(3^depth)
/// Space complexity: O(3^depth)
pub fn generateSierpinskiTriangles(
    allocator: std.mem.Allocator,
    vertex1: Point,
    vertex2: Point,
    vertex3: Point,
    depth: usize,
) ![]Triangle {
    var triangles = std.ArrayListUnmanaged(Triangle){};
    defer triangles.deinit(allocator);

    try buildTriangles(&triangles, allocator, vertex1, vertex2, vertex3, depth);
    return triangles.toOwnedSlice(allocator);
}

fn triangleCount(depth: usize) usize {
    var total: usize = 0;
    var power: usize = 1;
    for (0..depth + 1) |_| {
        total += power;
        power *= 3;
    }
    return total;
}

test "sierpinski triangle: python midpoint doctests" {
    const m1 = getMid(.{ .x = 0, .y = 0 }, .{ .x = 2, .y = 2 });
    try testing.expectApproxEqAbs(@as(f64, 1.0), m1.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m1.y, 1e-12);

    const m2 = getMid(.{ .x = -3, .y = -3 }, .{ .x = 3, .y = 3 });
    try testing.expectApproxEqAbs(@as(f64, 0.0), m2.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m2.y, 1e-12);

    const m3 = getMid(.{ .x = 1, .y = 0 }, .{ .x = 3, .y = 2 });
    try testing.expectApproxEqAbs(@as(f64, 2.0), m3.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m3.y, 1e-12);
}

test "sierpinski triangle: depth 0 and depth 1 geometry" {
    const alloc = testing.allocator;
    const a = Point{ .x = 0.0, .y = 0.0 };
    const b = Point{ .x = 2.0, .y = 0.0 };
    const c = Point{ .x = 1.0, .y = 2.0 };

    const d0 = try generateSierpinskiTriangles(alloc, a, b, c, 0);
    defer alloc.free(d0);
    try testing.expectEqual(@as(usize, 1), d0.len);
    try testing.expectApproxEqAbs(@as(f64, 0.0), d0[0].v1.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), d0[0].v2.x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0), d0[0].v3.y, 1e-12);

    const d1 = try generateSierpinskiTriangles(alloc, a, b, c, 1);
    defer alloc.free(d1);
    try testing.expectEqual(@as(usize, 4), d1.len);

    const expected = [_]Triangle{
        .{ .v1 = a, .v2 = b, .v3 = c },
        .{ .v1 = a, .v2 = .{ .x = 1.0, .y = 0.0 }, .v3 = .{ .x = 0.5, .y = 1.0 } },
        .{ .v1 = b, .v2 = .{ .x = 1.0, .y = 0.0 }, .v3 = .{ .x = 1.5, .y = 1.0 } },
        .{ .v1 = c, .v2 = .{ .x = 1.5, .y = 1.0 }, .v3 = .{ .x = 0.5, .y = 1.0 } },
    };

    for (expected, 0..) |triangle, i| {
        try testing.expectApproxEqAbs(triangle.v1.x, d1[i].v1.x, 1e-12);
        try testing.expectApproxEqAbs(triangle.v1.y, d1[i].v1.y, 1e-12);
        try testing.expectApproxEqAbs(triangle.v2.x, d1[i].v2.x, 1e-12);
        try testing.expectApproxEqAbs(triangle.v2.y, d1[i].v2.y, 1e-12);
        try testing.expectApproxEqAbs(triangle.v3.x, d1[i].v3.x, 1e-12);
        try testing.expectApproxEqAbs(triangle.v3.y, d1[i].v3.y, 1e-12);
    }
}

test "sierpinski triangle: recursive count and extreme depth" {
    const alloc = testing.allocator;

    const depth_5 = try generateSierpinskiTriangles(
        alloc,
        .{ .x = -175, .y = -125 },
        .{ .x = 0, .y = 175 },
        .{ .x = 175, .y = -125 },
        5,
    );
    defer alloc.free(depth_5);
    try testing.expectEqual(triangleCount(5), depth_5.len);

    const depth_8 = try generateSierpinskiTriangles(
        alloc,
        .{ .x = -175, .y = -125 },
        .{ .x = 0, .y = 175 },
        .{ .x = 175, .y = -125 },
        8,
    );
    defer alloc.free(depth_8);
    try testing.expectEqual(triangleCount(8), depth_8.len);

    try testing.expect(std.math.isFinite(depth_8[4_000].v1.x));
    try testing.expect(std.math.isFinite(depth_8[4_000].v2.y));
    try testing.expect(std.math.isFinite(depth_8[4_000].v3.x));
}
