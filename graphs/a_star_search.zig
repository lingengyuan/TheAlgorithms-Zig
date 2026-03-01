//! A* Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/a_star.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Edge = struct {
    to: usize,
    weight: u64,
};

pub const AStarResult = struct {
    path: []usize,
    cost: u64,
};

/// Finds a shortest path from `start` to `goal` using A* search on a weighted directed graph.
/// `adj` is adjacency-list based; invalid neighbor indices are ignored.
/// `heuristics[i]` must be non-negative, consistent across edges, and `heuristics[goal] == 0`.
/// Returns the found path (including start and goal) and path cost.
/// Time complexity: O(V² + E), Space complexity: O(V)
pub fn aStarSearch(
    allocator: Allocator,
    adj: []const []const Edge,
    heuristics: []const u64,
    start: usize,
    goal: usize,
) !AStarResult {
    const n = adj.len;
    if (start >= n or goal >= n) return error.InvalidNode;
    if (heuristics.len != n) return error.InvalidHeuristicLength;
    if (heuristics[goal] != 0) return error.InvalidGoalHeuristic;
    try validateConsistentHeuristic(adj, heuristics);

    const inf = std.math.maxInt(u64);
    const none = std.math.maxInt(usize);

    const g_score = try allocator.alloc(u64, n);
    defer allocator.free(g_score);
    const f_score = try allocator.alloc(u64, n);
    defer allocator.free(f_score);
    const came_from = try allocator.alloc(usize, n);
    defer allocator.free(came_from);
    const in_open = try allocator.alloc(bool, n);
    defer allocator.free(in_open);
    const closed = try allocator.alloc(bool, n);
    defer allocator.free(closed);

    @memset(g_score, inf);
    @memset(f_score, inf);
    @memset(came_from, none);
    @memset(in_open, false);
    @memset(closed, false);

    g_score[start] = 0;
    f_score[start] = heuristics[start];
    in_open[start] = true;

    while (true) {
        var best_idx: ?usize = null;
        var best_f: u64 = inf;
        var best_g: u64 = inf;

        for (0..n) |i| {
            if (!in_open[i]) continue;
            if (f_score[i] < best_f or (f_score[i] == best_f and g_score[i] < best_g)) {
                best_f = f_score[i];
                best_g = g_score[i];
                best_idx = i;
            }
        }

        const current = best_idx orelse return error.NoPath;

        if (current == goal) {
            const path = try reconstructPath(allocator, came_from, start, goal);
            return .{
                .path = path,
                .cost = g_score[goal],
            };
        }

        in_open[current] = false;
        closed[current] = true;

        const base = g_score[current];
        if (base == inf) continue;

        for (adj[current]) |edge| {
            if (edge.to >= n) continue;
            if (closed[edge.to]) continue;

            const add_g = @addWithOverflow(base, edge.weight);
            if (add_g[1] != 0) continue;
            const tentative_g = add_g[0];

            if (tentative_g < g_score[edge.to]) {
                came_from[edge.to] = current;
                g_score[edge.to] = tentative_g;

                const add_f = @addWithOverflow(tentative_g, heuristics[edge.to]);
                f_score[edge.to] = if (add_f[1] != 0) inf else add_f[0];
                in_open[edge.to] = true;
            }
        }
    }
}

fn validateConsistentHeuristic(adj: []const []const Edge, heuristics: []const u64) !void {
    for (adj, 0..) |edges, u| {
        const hu = heuristics[u];
        for (edges) |edge| {
            if (edge.to >= adj.len) continue;
            const with_overflow = @addWithOverflow(edge.weight, heuristics[edge.to]);
            if (with_overflow[1] != 0) continue;
            if (hu > with_overflow[0]) return error.InconsistentHeuristic;
        }
    }
}

fn reconstructPath(allocator: Allocator, came_from: []const usize, start: usize, goal: usize) ![]usize {
    const none = std.math.maxInt(usize);

    var len: usize = 1;
    var node = goal;
    while (node != start) {
        const parent = came_from[node];
        if (parent == none) return error.NoPath;
        node = parent;
        len += 1;
    }

    const path = try allocator.alloc(usize, len);
    var idx = len;
    node = goal;
    while (true) {
        idx -= 1;
        path[idx] = node;
        if (node == start) break;
        node = came_from[node];
    }

    return path;
}

test "a star search: basic shortest path" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 1 }, .{ .to = 2, .weight = 4 } },
        &[_]Edge{ .{ .to = 2, .weight = 2 }, .{ .to = 3, .weight = 5 } },
        &[_]Edge{.{ .to = 3, .weight = 1 }},
        &[_]Edge{.{ .to = 4, .weight = 3 }},
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 7, 6, 4, 3, 0 };

    const result = try aStarSearch(alloc, &adj, &heuristics, 0, 4);
    defer alloc.free(result.path);

    try testing.expectEqual(@as(u64, 7), result.cost);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3, 4 }, result.path);
}

test "a star search: start equals goal" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{},
    };
    const heuristics = [_]u64{0};

    const result = try aStarSearch(alloc, &adj, &heuristics, 0, 0);
    defer alloc.free(result.path);

    try testing.expectEqual(@as(u64, 0), result.cost);
    try testing.expectEqualSlices(usize, &[_]usize{0}, result.path);
}

test "a star search: unreachable goal returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{.{ .to = 1, .weight = 1 }},
        &[_]Edge{},
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 2, 1, 0 };

    try testing.expectError(error.NoPath, aStarSearch(alloc, &adj, &heuristics, 0, 2));
}

test "a star search: invalid node returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{},
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 0, 0 };

    try testing.expectError(error.InvalidNode, aStarSearch(alloc, &adj, &heuristics, 2, 1));
    try testing.expectError(error.InvalidNode, aStarSearch(alloc, &adj, &heuristics, 0, 2));
}

test "a star search: heuristic length mismatch returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{},
        &[_]Edge{},
    };
    const heuristics = [_]u64{0};

    try testing.expectError(error.InvalidHeuristicLength, aStarSearch(alloc, &adj, &heuristics, 0, 1));
}

test "a star search: goal heuristic must be zero" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{.{ .to = 1, .weight = 1 }},
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 1, 2 };

    try testing.expectError(error.InvalidGoalHeuristic, aStarSearch(alloc, &adj, &heuristics, 0, 1));
}

test "a star search: inconsistent heuristic returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 2 }, .{ .to = 2, .weight = 1 } },
        &[_]Edge{.{ .to = 3, .weight = 1 }},
        &[_]Edge{ .{ .to = 1, .weight = 0 }, .{ .to = 3, .weight = 100 } },
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 0, 0, 100, 0 };

    try testing.expectError(error.InconsistentHeuristic, aStarSearch(alloc, &adj, &heuristics, 0, 3));
}

test "a star search: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 2 }, .{ .to = 99, .weight = 1 } },
        &[_]Edge{.{ .to = 2, .weight = 2 }},
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 3, 1, 0 };

    const result = try aStarSearch(alloc, &adj, &heuristics, 0, 2);
    defer alloc.free(result.path);

    try testing.expectEqual(@as(u64, 4), result.cost);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, result.path);
}

test "a star search: overflow-prone path is skipped" {
    const alloc = testing.allocator;
    const max_u64 = std.math.maxInt(u64);
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 5 }, .{ .to = 2, .weight = 6 } },
        &[_]Edge{.{ .to = 3, .weight = max_u64 }},
        &[_]Edge{.{ .to = 3, .weight = 7 }},
        &[_]Edge{},
    };
    const heuristics = [_]u64{ 13, 8, 7, 0 };

    const result = try aStarSearch(alloc, &adj, &heuristics, 0, 3);
    defer alloc.free(result.path);

    try testing.expectEqual(@as(u64, 13), result.cost);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 2, 3 }, result.path);
}

test "a star search: empty graph returns invalid node" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{};
    const heuristics = [_]u64{};

    try testing.expectError(error.InvalidNode, aStarSearch(alloc, &adj, &heuristics, 0, 0));
}
