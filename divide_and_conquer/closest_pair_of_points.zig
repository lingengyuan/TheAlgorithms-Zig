//! Closest Pair of Points (Divide and Conquer) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/closest_pair_of_points.py

const std = @import("std");
const testing = std.testing;

pub const Point = struct {
    x: f64,
    y: f64,
};

pub const ClosestPairError = error{ NotEnoughPoints, InvalidCount, InvalidColumn };

fn lessByX(_: void, a: Point, b: Point) bool {
    return if (a.x == b.x) a.y < b.y else a.x < b.x;
}

fn lessByY(_: void, a: Point, b: Point) bool {
    return if (a.y == b.y) a.x < b.x else a.y < b.y;
}

/// Squared Euclidean distance between two points.
pub fn euclideanDistanceSqr(point1: Point, point2: Point) f64 {
    const dx = point1.x - point2.x;
    const dy = point1.y - point2.y;
    return dx * dx + dy * dy;
}

/// Returns a copy sorted by selected column (0 => x, 1 => y).
pub fn columnBasedSort(
    allocator: std.mem.Allocator,
    points: []const Point,
    column: u8,
) (ClosestPairError || std.mem.Allocator.Error)![]Point {
    if (column != 0 and column != 1) return ClosestPairError.InvalidColumn;

    const out = try allocator.dupe(Point, points);
    if (column == 0) {
        std.mem.sort(Point, out, {}, lessByX);
    } else {
        std.mem.sort(Point, out, {}, lessByY);
    }
    return out;
}

/// Brute-force closest-pair squared distance for small point sets.
pub fn disBetweenClosestPair(points: []const Point, points_count: usize, min_dis: f64) f64 {
    var best = min_dis;
    if (points_count < 2) return best;

    var i: usize = 0;
    while (i + 1 < points_count) : (i += 1) {
        var j = i + 1;
        while (j < points_count) : (j += 1) {
            best = @min(best, euclideanDistanceSqr(points[i], points[j]));
        }
    }
    return best;
}

/// Closest-pair squared distance within a y-sorted strip.
pub fn disBetweenClosestInStrip(points: []const Point, points_count: usize, min_dis: f64) f64 {
    var best = min_dis;

    var i: usize = 0;
    while (i < points_count) : (i += 1) {
        var j = i + 1;
        while (j < points_count and j <= i + 7) : (j += 1) {
            const dy = points[j].y - points[i].y;
            if (dy * dy >= best) break;
            best = @min(best, euclideanDistanceSqr(points[i], points[j]));
        }
    }
    return best;
}

fn closestPairOfPointsSqr(
    allocator: std.mem.Allocator,
    points_sorted_on_x: []const Point,
    points_sorted_on_y: []const Point,
    points_count: usize,
) std.mem.Allocator.Error!f64 {
    if (points_count <= 3) {
        return disBetweenClosestPair(points_sorted_on_x, points_count, std.math.inf(f64));
    }

    const mid = points_count / 2;
    const mid_point = points_sorted_on_x[mid];

    const left_x = points_sorted_on_x[0..mid];
    const right_x = points_sorted_on_x[mid..points_count];

    var left_y = try allocator.alloc(Point, left_x.len);
    defer allocator.free(left_y);
    var right_y = try allocator.alloc(Point, right_x.len);
    defer allocator.free(right_y);

    var li: usize = 0;
    var ri: usize = 0;
    for (points_sorted_on_y) |p| {
        if ((p.x < mid_point.x or (p.x == mid_point.x and li < left_x.len)) and li < left_x.len) {
            left_y[li] = p;
            li += 1;
        } else if (ri < right_x.len) {
            right_y[ri] = p;
            ri += 1;
        }
    }

    const closest_in_left = try closestPairOfPointsSqr(allocator, left_x, left_y[0..li], left_x.len);
    const closest_in_right = try closestPairOfPointsSqr(allocator, right_x, right_y[0..ri], right_x.len);
    const closest_pair_dis = @min(closest_in_left, closest_in_right);

    var strip = std.ArrayListUnmanaged(Point){};
    defer strip.deinit(allocator);

    for (points_sorted_on_y) |p| {
        const dx = p.x - mid_point.x;
        if (dx * dx < closest_pair_dis) {
            try strip.append(allocator, p);
        }
    }

    const closest_in_strip = disBetweenClosestInStrip(strip.items, strip.items.len, closest_pair_dis);
    return @min(closest_pair_dis, closest_in_strip);
}

/// Returns minimum Euclidean distance among input points.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n)
pub fn closestPairOfPoints(
    allocator: std.mem.Allocator,
    points: []const Point,
    points_count: usize,
) (ClosestPairError || std.mem.Allocator.Error)!f64 {
    if (points_count != points.len) return ClosestPairError.InvalidCount;
    if (points_count < 2) return ClosestPairError.NotEnoughPoints;

    const points_sorted_on_x = try columnBasedSort(allocator, points, 0);
    defer allocator.free(points_sorted_on_x);
    const points_sorted_on_y = try columnBasedSort(allocator, points, 1);
    defer allocator.free(points_sorted_on_y);

    const best_sqr = try closestPairOfPointsSqr(allocator, points_sorted_on_x, points_sorted_on_y, points_count);
    return @sqrt(best_sqr);
}

test "closest pair: helper examples" {
    try testing.expectEqual(@as(f64, 5), euclideanDistanceSqr(.{ .x = 1, .y = 2 }, .{ .x = 2, .y = 4 }));

    const alloc = testing.allocator;
    const points = [_]Point{
        .{ .x = 5, .y = 1 },
        .{ .x = 4, .y = 2 },
        .{ .x = 3, .y = 0 },
    };

    const sorted = try columnBasedSort(alloc, &points, 1);
    defer alloc.free(sorted);
    try testing.expectApproxEqAbs(@as(f64, 3), sorted[0].x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), sorted[0].y, 1e-12);
}

test "closest pair: python examples" {
    const alloc = testing.allocator;

    const points1 = [_]Point{
        .{ .x = 2, .y = 3 },
        .{ .x = 12, .y = 30 },
    };
    const d1 = try closestPairOfPoints(alloc, &points1, points1.len);
    try testing.expectApproxEqAbs(@as(f64, 28.792360097775937), d1, 1e-12);

    const points2 = [_]Point{
        .{ .x = 2, .y = 3 },
        .{ .x = 12, .y = 30 },
        .{ .x = 40, .y = 50 },
        .{ .x = 5, .y = 1 },
        .{ .x = 12, .y = 10 },
        .{ .x = 3, .y = 4 },
    };
    const d2 = try closestPairOfPoints(alloc, &points2, points2.len);
    try testing.expectApproxEqAbs(@sqrt(@as(f64, 2)), d2, 1e-12);
}

test "closest pair: validation and extreme" {
    const alloc = testing.allocator;

    const one = [_]Point{.{ .x = 0, .y = 0 }};
    try testing.expectError(ClosestPairError.NotEnoughPoints, closestPairOfPoints(alloc, &one, one.len));
    try testing.expectError(ClosestPairError.InvalidCount, closestPairOfPoints(alloc, &one, 2));

    var points: [256]Point = undefined;
    for (0..points.len) |i| {
        const x = @as(f64, @floatFromInt(i));
        points[i] = .{ .x = x, .y = 2.0 * x };
    }
    const d = try closestPairOfPoints(alloc, points[0..], points.len);
    try testing.expectApproxEqAbs(@sqrt(@as(f64, 5)), d, 1e-12);
}
