//! Bidirectional Dijkstra - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/bi_directional_dijkstra.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const WeightedEdge = struct {
    to: usize,
    weight: i64,
};

/// Computes shortest path distance from `source` to `destination` using bidirectional Dijkstra.
/// `graph_forward` and `graph_backward` must represent forward and reverse edges on same node set.
/// Invalid neighbor indices are ignored. Returns `null` if unreachable.
/// Time complexity: O(V^2 + E), Space complexity: O(V)
pub fn bidirectionalDijkstra(
    allocator: Allocator,
    graph_forward: []const []const WeightedEdge,
    graph_backward: []const []const WeightedEdge,
    source: usize,
    destination: usize,
) !?i64 {
    const n = graph_forward.len;
    if (n != graph_backward.len) return error.InvalidGraph;
    if (source >= n or destination >= n) return error.InvalidNode;
    if (source == destination) return 0;

    const inf: i64 = std.math.maxInt(i64);

    const dist_fwd = try allocator.alloc(i64, n);
    defer allocator.free(dist_fwd);
    const dist_bwd = try allocator.alloc(i64, n);
    defer allocator.free(dist_bwd);
    const visited_fwd = try allocator.alloc(bool, n);
    defer allocator.free(visited_fwd);
    const visited_bwd = try allocator.alloc(bool, n);
    defer allocator.free(visited_bwd);

    @memset(dist_fwd, inf);
    @memset(dist_bwd, inf);
    @memset(visited_fwd, false);
    @memset(visited_bwd, false);

    dist_fwd[source] = 0;
    dist_bwd[destination] = 0;

    var shortest: i64 = inf;

    while (true) {
        const v_fwd_opt = minUnvisited(dist_fwd, visited_fwd);
        const v_bwd_opt = minUnvisited(dist_bwd, visited_bwd);
        if (v_fwd_opt == null or v_bwd_opt == null) break;

        const v_fwd = v_fwd_opt.?;
        const v_bwd = v_bwd_opt.?;
        visited_fwd[v_fwd] = true;
        visited_bwd[v_bwd] = true;

        try passAndRelax(
            graph_forward,
            v_fwd,
            visited_fwd,
            visited_bwd,
            dist_fwd,
            dist_bwd,
            &shortest,
        );
        try passAndRelax(
            graph_backward,
            v_bwd,
            visited_bwd,
            visited_fwd,
            dist_bwd,
            dist_fwd,
            &shortest,
        );

        if (shortest != inf) {
            const sum = @addWithOverflow(dist_fwd[v_fwd], dist_bwd[v_bwd]);
            if (sum[1] == 0 and sum[0] >= shortest) break;
        }
    }

    if (shortest == inf) return null;
    return shortest;
}

fn minUnvisited(dist: []const i64, visited: []const bool) ?usize {
    const inf = std.math.maxInt(i64);
    var best_idx: ?usize = null;
    var best_cost: i64 = inf;

    for (0..dist.len) |i| {
        if (visited[i]) continue;
        if (dist[i] < best_cost) {
            best_cost = dist[i];
            best_idx = i;
        }
    }
    if (best_cost == inf) return null;
    return best_idx;
}

fn passAndRelax(
    graph: []const []const WeightedEdge,
    v: usize,
    visited_forward: []const bool,
    visited_backward: []const bool,
    dist_forward: []i64,
    dist_backward: []const i64,
    shortest: *i64,
) !void {
    if (dist_forward[v] == std.math.maxInt(i64)) return;
    for (graph[v]) |edge| {
        if (edge.weight < 0) return error.NegativeWeight;
        if (edge.to >= graph.len) continue;
        if (visited_forward[edge.to]) continue;

        const direct = @addWithOverflow(dist_forward[v], edge.weight);
        if (direct[1] != 0) return error.Overflow;
        const new_cost = direct[0];
        if (new_cost < dist_forward[edge.to]) {
            dist_forward[edge.to] = new_cost;
        }

        if (visited_backward[edge.to] and dist_backward[edge.to] != std.math.maxInt(i64)) {
            const via_sum = @addWithOverflow(new_cost, dist_backward[edge.to]);
            if (via_sum[1] != 0) return error.Overflow;
            if (via_sum[0] < shortest.*) shortest.* = via_sum[0];
        }
    }
}

test "bidirectional dijkstra: python sample graph distance" {
    const alloc = testing.allocator;

    // B=0, C=1, D=2, E=3, F=4, G=5
    const graph_fwd = [_][]const WeightedEdge{
        &[_]WeightedEdge{.{ .to = 1, .weight = 1 }}, // B
        &[_]WeightedEdge{.{ .to = 2, .weight = 1 }}, // C
        &[_]WeightedEdge{.{ .to = 4, .weight = 1 }}, // D
        &[_]WeightedEdge{ .{ .to = 0, .weight = 1 }, .{ .to = 5, .weight = 2 } }, // E
        &[_]WeightedEdge{}, // F
        &[_]WeightedEdge{.{ .to = 4, .weight = 1 }}, // G
    };

    const graph_bwd = [_][]const WeightedEdge{
        &[_]WeightedEdge{.{ .to = 3, .weight = 1 }}, // B <- E
        &[_]WeightedEdge{.{ .to = 0, .weight = 1 }}, // C <- B
        &[_]WeightedEdge{.{ .to = 1, .weight = 1 }}, // D <- C
        &[_]WeightedEdge{}, // E
        &[_]WeightedEdge{ .{ .to = 2, .weight = 1 }, .{ .to = 5, .weight = 1 } }, // F <- D,G
        &[_]WeightedEdge{.{ .to = 3, .weight = 2 }}, // G <- E
    };

    const distance = (try bidirectionalDijkstra(alloc, &graph_fwd, &graph_bwd, 3, 4)).?;
    try testing.expectEqual(@as(i64, 3), distance);
}

test "bidirectional dijkstra: unreachable and same node" {
    const alloc = testing.allocator;
    const graph = [_][]const WeightedEdge{
        &[_]WeightedEdge{.{ .to = 1, .weight = 1 }},
        &[_]WeightedEdge{},
        &[_]WeightedEdge{},
    };

    try testing.expectEqual(@as(?i64, 0), try bidirectionalDijkstra(alloc, &graph, &graph, 1, 1));
    try testing.expectEqual(@as(?i64, null), try bidirectionalDijkstra(alloc, &graph, &graph, 0, 2));
}

test "bidirectional dijkstra: invalid and negative inputs" {
    const alloc = testing.allocator;
    const good = [_][]const WeightedEdge{
        &[_]WeightedEdge{.{ .to = 1, .weight = 1 }},
        &[_]WeightedEdge{},
    };
    const bad_reverse = [_][]const WeightedEdge{
        &[_]WeightedEdge{},
    };
    try testing.expectError(error.InvalidGraph, bidirectionalDijkstra(alloc, &good, &bad_reverse, 0, 1));
    try testing.expectError(error.InvalidNode, bidirectionalDijkstra(alloc, &good, &good, 5, 1));

    const negative = [_][]const WeightedEdge{
        &[_]WeightedEdge{.{ .to = 1, .weight = -1 }},
        &[_]WeightedEdge{},
    };
    try testing.expectError(error.NegativeWeight, bidirectionalDijkstra(alloc, &negative, &negative, 0, 1));
}

test "bidirectional dijkstra: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 256;

    const fwd_mut = try alloc.alloc([]WeightedEdge, n);
    defer {
        for (fwd_mut) |row| alloc.free(row);
        alloc.free(fwd_mut);
    }
    const bwd_mut = try alloc.alloc([]WeightedEdge, n);
    defer {
        for (bwd_mut) |row| alloc.free(row);
        alloc.free(bwd_mut);
    }

    for (0..n) |i| {
        if (i + 1 < n) {
            fwd_mut[i] = try alloc.alloc(WeightedEdge, 1);
            fwd_mut[i][0] = .{ .to = i + 1, .weight = 1 };
        } else {
            fwd_mut[i] = try alloc.alloc(WeightedEdge, 0);
        }

        if (i > 0) {
            bwd_mut[i] = try alloc.alloc(WeightedEdge, 1);
            bwd_mut[i][0] = .{ .to = i - 1, .weight = 1 };
        } else {
            bwd_mut[i] = try alloc.alloc(WeightedEdge, 0);
        }
    }

    const graph_fwd = try alloc.alloc([]const WeightedEdge, n);
    defer alloc.free(graph_fwd);
    const graph_bwd = try alloc.alloc([]const WeightedEdge, n);
    defer alloc.free(graph_bwd);

    for (0..n) |i| {
        graph_fwd[i] = fwd_mut[i];
        graph_bwd[i] = bwd_mut[i];
    }

    const distance = (try bidirectionalDijkstra(alloc, graph_fwd, graph_bwd, 0, n - 1)).?;
    try testing.expectEqual(@as(i64, @intCast(n - 1)), distance);
}
