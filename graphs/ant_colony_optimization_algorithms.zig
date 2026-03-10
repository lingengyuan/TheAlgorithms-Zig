//! Ant Colony Optimization for TSP - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/ant_colony_optimization_algorithms.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Point = struct {
    x: f64,
    y: f64,
};

pub const RouteResult = struct {
    path: []usize,
    distance: f64,

    pub fn deinit(self: RouteResult, allocator: Allocator) void {
        allocator.free(self.path);
    }
};

/// Computes Euclidean distance between two points.
/// Time complexity: O(1), Space complexity: O(1)
pub fn distance(a: Point, b: Point) f64 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return @sqrt(dx * dx + dy * dy);
}

/// Selects the next city using the Python scoring rule.
/// When `random` is `null`, ties are resolved deterministically by maximum score then
/// lowest city index to keep tests stable.
/// Time complexity: O(n), Space complexity: O(n)
pub fn citySelect(
    pheromone: []const []const f64,
    cities: []const Point,
    current_city: usize,
    unvisited: []const usize,
    alpha: f64,
    beta: f64,
    random: ?std.Random,
) !usize {
    if (pheromone.len == 0) return error.InvalidPheromone;
    if (cities.len == 0) return error.EmptyCities;
    if (current_city >= cities.len) return error.InvalidCity;
    if (unvisited.len == 0) return error.NoUnvisitedCities;
    if (pheromone.len != cities.len) return error.InvalidPheromone;
    for (pheromone) |row| {
        if (row.len != cities.len) return error.InvalidPheromone;
    }

    var best_idx: usize = 0;
    var best_score: f64 = -1.0;
    var total_weight: f64 = 0;

    for (unvisited, 0..) |city, i| {
        if (city >= cities.len) return error.InvalidCity;
        const city_distance = distance(cities[city], cities[current_city]);
        if (city_distance == 0) return error.ZeroDistance;
        const score = std.math.pow(f64, pheromone[city][current_city], alpha) *
            std.math.pow(f64, 1.0 / city_distance, beta);
        if (score > best_score or (score == best_score and city < unvisited[best_idx])) {
            best_score = score;
            best_idx = i;
        }
        total_weight += score;
    }

    if (random) |rng| {
        if (total_weight > 0) {
            var threshold = rng.float(f64) * total_weight;
            for (unvisited, 0..) |city, i| {
                const city_distance = distance(cities[city], cities[current_city]);
                const score = std.math.pow(f64, pheromone[city][current_city], alpha) *
                    std.math.pow(f64, 1.0 / city_distance, beta);
                threshold -= score;
                if (threshold <= 0) return unvisited[i];
            }
        }
    }

    return unvisited[best_idx];
}

/// Updates the pheromone matrix and best route after one iteration.
/// Time complexity: O(n² + ants * route_len), Space complexity: O(1) extra besides matrix
pub fn pheromoneUpdate(
    pheromone: []const []f64,
    cities: []const Point,
    pheromone_evaporation: f64,
    ant_routes: []const []const usize,
    q: f64,
    best_path: []const usize,
    best_distance: f64,
    allocator: Allocator,
) !RouteResult {
    if (cities.len == 0) return error.EmptyCities;
    if (pheromone.len != cities.len) return error.InvalidPheromone;
    for (pheromone) |row| {
        if (row.len != cities.len) return error.InvalidPheromone;
    }

    for (0..cities.len) |a| {
        for (0..cities.len) |b| {
            pheromone[a][b] *= pheromone_evaporation;
        }
    }

    var best_result = RouteResult{
        .path = try allocator.dupe(usize, best_path),
        .distance = best_distance,
    };
    errdefer best_result.deinit(allocator);

    for (ant_routes) |route| {
        if (route.len < 2) continue;
        var total_distance: f64 = 0;
        for (route[0 .. route.len - 1], 0..) |from, i| {
            const to = route[i + 1];
            if (from >= cities.len or to >= cities.len) return error.InvalidCity;
            total_distance += distance(cities[from], cities[to]);
        }
        if (total_distance == 0) return error.ZeroDistance;

        const delta_pheromone = q / total_distance;
        for (route[0 .. route.len - 1], 0..) |from, i| {
            const to = route[i + 1];
            pheromone[from][to] += delta_pheromone;
            pheromone[to][from] = pheromone[from][to];
        }

        if (total_distance < best_result.distance) {
            allocator.free(best_result.path);
            best_result.path = try allocator.dupe(usize, route);
            best_result.distance = total_distance;
        }
    }

    return best_result;
}

/// Solves a TSP-like route using ant colony optimization.
/// By default the implementation uses deterministic tie-breaking for repeatable tests.
/// Returns empty path and +inf when `ants_num == 0` or `iterations_num == 0`.
/// Time complexity: O(iterations * ants * n²), Space complexity: O(n²)
pub fn solveAntColonyTsp(
    allocator: Allocator,
    cities: []const Point,
    ants_num: usize,
    iterations_num: usize,
    pheromone_evaporation: f64,
    alpha: f64,
    beta: f64,
    q: f64,
    random: ?std.Random,
) !RouteResult {
    if (cities.len == 0) return error.EmptyCities;
    if (ants_num == 0 or iterations_num == 0) {
        return .{
            .path = try allocator.alloc(usize, 0),
            .distance = std.math.inf(f64),
        };
    }

    const pheromone_data = try allocator.alloc(f64, cities.len * cities.len);
    defer allocator.free(pheromone_data);
    @memset(pheromone_data, 1.0);

    const pheromone_rows = try allocator.alloc([]f64, cities.len);
    defer allocator.free(pheromone_rows);
    for (0..cities.len) |i| {
        pheromone_rows[i] = pheromone_data[i * cities.len .. (i + 1) * cities.len];
    }

    var best = RouteResult{
        .path = try allocator.alloc(usize, 0),
        .distance = std.math.inf(f64),
    };
    errdefer best.deinit(allocator);

    var routes_storage = std.ArrayListUnmanaged([]usize){};
    defer routes_storage.deinit(allocator);

    for (0..iterations_num) |_| {
        for (routes_storage.items) |route| allocator.free(route);
        routes_storage.clearRetainingCapacity();

        for (0..ants_num) |_| {
            const route = try buildAntRoute(allocator, pheromone_rows, cities, alpha, beta, random);
            try routes_storage.append(allocator, route);
        }

        const updated = try pheromoneUpdate(
            pheromone_rows,
            cities,
            pheromone_evaporation,
            routes_storage.items,
            q,
            best.path,
            best.distance,
            allocator,
        );
        best.deinit(allocator);
        best = updated;
    }

    for (routes_storage.items) |route| allocator.free(route);
    return best;
}

fn buildAntRoute(
    allocator: Allocator,
    pheromone: []const []const f64,
    cities: []const Point,
    alpha: f64,
    beta: f64,
    random: ?std.Random,
) ![]usize {
    var route = std.ArrayListUnmanaged(usize){};
    defer route.deinit(allocator);
    try route.append(allocator, 0);

    const unvisited_flags = try allocator.alloc(bool, cities.len);
    defer allocator.free(unvisited_flags);
    @memset(unvisited_flags, true);
    unvisited_flags[0] = false;

    var current: usize = 0;
    while (true) {
        var remaining_count: usize = 0;
        for (unvisited_flags) |flag| {
            if (flag) remaining_count += 1;
        }
        if (remaining_count == 0) break;

        const unvisited = try allocator.alloc(usize, remaining_count);
        defer allocator.free(unvisited);
        var write_idx: usize = 0;
        for (unvisited_flags, 0..) |flag, city| {
            if (flag) {
                unvisited[write_idx] = city;
                write_idx += 1;
            }
        }

        const next = try citySelect(pheromone, cities, current, unvisited, alpha, beta, random);
        try route.append(allocator, next);
        unvisited_flags[next] = false;
        current = next;
    }

    try route.append(allocator, 0);
    return try route.toOwnedSlice(allocator);
}

test "ant colony optimization: distance helper" {
    try testing.expectApproxEqAbs(@as(f64, 5.0), distance(.{ .x = 0, .y = 0 }, .{ .x = 3, .y = 4 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0), distance(.{ .x = 0, .y = 0 }, .{ .x = -3, .y = 4 }), 1e-12);
}

test "ant colony optimization: city select with one choice" {
    const cities = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 2 },
    };
    const pheromone = [_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 1.0, 1.0 },
    };

    try testing.expectEqual(@as(usize, 1), try citySelect(&pheromone, &cities, 0, &[_]usize{1}, 1.0, 5.0, null));
}

test "ant colony optimization: pheromone update python sample" {
    const alloc = testing.allocator;
    const cities = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 2 },
    };

    const data = try alloc.alloc(f64, 4);
    defer alloc.free(data);
    data[0] = 1.0;
    data[1] = 1.0;
    data[2] = 1.0;
    data[3] = 1.0;
    const pheromone = [_][]f64{
        data[0..2],
        data[2..4],
    };
    const ant_route = [_][]const usize{
        &[_]usize{ 0, 1, 0 },
    };

    var result = try pheromoneUpdate(pheromone[0..], &cities, 0.7, &ant_route, 10.0, &[_]usize{}, std.math.inf(f64), alloc);
    defer result.deinit(alloc);

    try testing.expectApproxEqAbs(@as(f64, 0.7), pheromone[0][0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.235533905932737), pheromone[0][1], 1e-9);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 0 }, result.path);
    try testing.expectApproxEqAbs(@as(f64, 5.656854249492381), result.distance, 1e-9);
}

test "ant colony optimization: sample route deterministic" {
    const alloc = testing.allocator;
    const cities = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 0, .y = 5 },
        .{ .x = 3, .y = 8 },
        .{ .x = 8, .y = 10 },
        .{ .x = 12, .y = 8 },
        .{ .x = 12, .y = 4 },
        .{ .x = 8, .y = 0 },
        .{ .x = 6, .y = 2 },
    };

    var result = try solveAntColonyTsp(alloc, &cities, 10, 20, 0.7, 1.0, 5.0, 10.0, null);
    defer result.deinit(alloc);

    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 0 }, result.path);
    try testing.expectApproxEqAbs(@as(f64, 37.909778143828696), result.distance, 1e-9);
}

test "ant colony optimization: empty and degenerate cases" {
    const alloc = testing.allocator;
    try testing.expectError(error.EmptyCities, solveAntColonyTsp(alloc, &[_]Point{}, 5, 5, 0.7, 1.0, 5.0, 10.0, null));

    const cities = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 2 },
    };
    var zero_ants = try solveAntColonyTsp(alloc, &cities, 0, 5, 0.7, 1.0, 5.0, 10.0, null);
    defer zero_ants.deinit(alloc);
    try testing.expectEqual(@as(usize, 0), zero_ants.path.len);
    try testing.expect(std.math.isInf(zero_ants.distance));

    var zero_iterations = try solveAntColonyTsp(alloc, &cities, 5, 0, 0.7, 1.0, 5.0, 10.0, null);
    defer zero_iterations.deinit(alloc);
    try testing.expectEqual(@as(usize, 0), zero_iterations.path.len);
    try testing.expect(std.math.isInf(zero_iterations.distance));
}

test "ant colony optimization: two city and extreme long ring" {
    const alloc = testing.allocator;
    const two = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 2 },
    };
    var small = try solveAntColonyTsp(alloc, &two, 5, 5, 0.7, 1.0, 5.0, 10.0, null);
    defer small.deinit(alloc);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 0 }, small.path);
    try testing.expectApproxEqAbs(@as(f64, 5.656854249492381), small.distance, 1e-9);

    const n: usize = 32;
    const points = try alloc.alloc(Point, n);
    defer alloc.free(points);
    for (0..n) |i| {
        const theta = (2.0 * std.math.pi * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(n));
        points[i] = .{ .x = @cos(theta) * 100.0, .y = @sin(theta) * 100.0 };
    }

    var large = try solveAntColonyTsp(alloc, points, 4, 6, 0.8, 1.0, 3.0, 5.0, null);
    defer large.deinit(alloc);
    try testing.expectEqual(n + 1, large.path.len);
    try testing.expectEqual(@as(usize, 0), large.path[0]);
    try testing.expectEqual(@as(usize, 0), large.path[large.path.len - 1]);
}
