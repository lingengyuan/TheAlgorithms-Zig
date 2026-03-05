//! Convex Hull - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/convex_hull.py

const std = @import("std");
const testing = std.testing;

pub const Point = struct {
    x: f64,
    y: f64,
};

pub const ConvexHullError = error{EmptyInput};

fn lessPoint(_: void, a: Point, b: Point) bool {
    return if (a.x == b.x) a.y < b.y else a.x < b.x;
}

fn pointEq(a: Point, b: Point) bool {
    return a.x == b.x and a.y == b.y;
}

fn cross(o: Point, a: Point, b: Point) f64 {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

fn dedupeSortedInPlace(points: []Point) usize {
    if (points.len == 0) return 0;
    var write: usize = 1;
    for (points[1..]) |p| {
        if (!pointEq(p, points[write - 1])) {
            points[write] = p;
            write += 1;
        }
    }
    return write;
}

/// Computes convex hull vertices and returns them sorted by `(x, y)`.
///
/// API note: uses monotonic chain and returns the hull vertex set sorted,
/// equivalent to Python examples for convex hull membership output.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n)
pub fn convexHull(
    allocator: std.mem.Allocator,
    points_input: []const Point,
) (ConvexHullError || std.mem.Allocator.Error)![]Point {
    if (points_input.len == 0) return ConvexHullError.EmptyInput;

    var points = try allocator.dupe(Point, points_input);
    defer allocator.free(points);
    std.mem.sort(Point, points, {}, lessPoint);
    const unique_len = dedupeSortedInPlace(points);
    const unique = points[0..unique_len];

    if (unique.len <= 2) return allocator.dupe(Point, unique);

    var lower = std.ArrayListUnmanaged(Point){};
    defer lower.deinit(allocator);
    for (unique) |p| {
        while (lower.items.len >= 2) {
            const l = lower.items.len;
            if (cross(lower.items[l - 2], lower.items[l - 1], p) <= 0) {
                _ = lower.pop();
            } else break;
        }
        try lower.append(allocator, p);
    }

    var upper = std.ArrayListUnmanaged(Point){};
    defer upper.deinit(allocator);
    var i = unique.len;
    while (i > 0) {
        i -= 1;
        const p = unique[i];
        while (upper.items.len >= 2) {
            const l = upper.items.len;
            if (cross(upper.items[l - 2], upper.items[l - 1], p) <= 0) {
                _ = upper.pop();
            } else break;
        }
        try upper.append(allocator, p);
    }

    var hull = std.ArrayListUnmanaged(Point){};
    defer hull.deinit(allocator);

    // skip last element of each chain to avoid duplicate endpoints
    for (0..lower.items.len - 1) |idx| try hull.append(allocator, lower.items[idx]);
    for (0..upper.items.len - 1) |idx| try hull.append(allocator, upper.items[idx]);

    std.mem.sort(Point, hull.items, {}, lessPoint);
    const dedup_hull_len = dedupeSortedInPlace(hull.items);
    return allocator.dupe(Point, hull.items[0..dedup_hull_len]);
}

fn expectPointsEq(expected: []const Point, got: []const Point, tol: f64) !void {
    try testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        try testing.expectApproxEqAbs(e.x, g.x, tol);
        try testing.expectApproxEqAbs(e.y, g.y, tol);
    }
}

test "convex hull: python examples" {
    const alloc = testing.allocator;

    const p1 = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 10, .y = 1 },
    };
    const h1 = try convexHull(alloc, &p1);
    defer alloc.free(h1);
    try expectPointsEq(&[_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 10, .y = 1 },
    }, h1, 1e-12);

    const p2 = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 10, .y = 0 },
    };
    const h2 = try convexHull(alloc, &p2);
    defer alloc.free(h2);
    try expectPointsEq(&[_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
    }, h2, 1e-12);

    const p3 = [_]Point{
        .{ .x = -1, .y = 1 },
        .{ .x = -1, .y = -1 },
        .{ .x = 0, .y = 0 },
        .{ .x = 0.5, .y = 0.5 },
        .{ .x = 1, .y = -1 },
        .{ .x = 1, .y = 1 },
        .{ .x = -0.75, .y = 1 },
    };
    const h3 = try convexHull(alloc, &p3);
    defer alloc.free(h3);
    try expectPointsEq(&[_]Point{
        .{ .x = -1, .y = -1 },
        .{ .x = -1, .y = 1 },
        .{ .x = 1, .y = -1 },
        .{ .x = 1, .y = 1 },
    }, h3, 1e-12);
}

test "convex hull: validation and extreme" {
    const alloc = testing.allocator;
    try testing.expectError(ConvexHullError.EmptyInput, convexHull(alloc, &[_]Point{}));

    var points: [1024]Point = undefined;
    for (0..points.len) |i| {
        const t = @as(f64, @floatFromInt(i)) * 0.03125;
        points[i] = .{ .x = @cos(t), .y = @sin(t) };
    }
    const hull = try convexHull(alloc, points[0..]);
    defer alloc.free(hull);
    try testing.expect(hull.len > 100);
}
