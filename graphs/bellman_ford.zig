//! Bellman-Ford Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/bellman_ford.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Edge = struct {
    from: usize,
    to: usize,
    weight: i64,
};

/// Computes single-source shortest path distances with support for negative edges.
/// Returns `error.NegativeCycle` if a reachable negative cycle exists.
/// Distances to unreachable nodes are `std.math.maxInt(i64)`.
/// Invalid edge endpoints are ignored.
/// Time complexity: O(V * E), Space complexity: O(V)
pub fn bellmanFord(allocator: Allocator, vertex_count: usize, edges: []const Edge, start: usize) ![]i64 {
    if (start >= vertex_count) return try allocator.alloc(i64, 0);

    const inf = std.math.maxInt(i64);
    const dist = try allocator.alloc(i64, vertex_count);
    @memset(dist, inf);
    dist[start] = 0;

    var pass: usize = 0;
    while (pass + 1 < vertex_count) : (pass += 1) {
        var changed = false;
        for (edges) |edge| {
            if (edge.from >= vertex_count or edge.to >= vertex_count) continue;
            if (dist[edge.from] == inf) continue;

            const sum = @addWithOverflow(dist[edge.from], edge.weight);
            if (sum[1] != 0) continue;

            if (sum[0] < dist[edge.to]) {
                dist[edge.to] = sum[0];
                changed = true;
            }
        }
        if (!changed) break;
    }

    for (edges) |edge| {
        if (edge.from >= vertex_count or edge.to >= vertex_count) continue;
        if (dist[edge.from] == inf) continue;

        const sum = @addWithOverflow(dist[edge.from], edge.weight);
        if (sum[1] != 0) continue;

        if (sum[0] < dist[edge.to]) {
            allocator.free(dist);
            return error.NegativeCycle;
        }
    }

    return dist;
}

test "bellman-ford: basic graph with negative edge" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .from = 0, .to = 1, .weight = -1 },
        .{ .from = 0, .to = 2, .weight = 4 },
        .{ .from = 1, .to = 2, .weight = 3 },
        .{ .from = 1, .to = 3, .weight = 2 },
        .{ .from = 1, .to = 4, .weight = 2 },
        .{ .from = 3, .to = 2, .weight = 5 },
        .{ .from = 3, .to = 1, .weight = 1 },
        .{ .from = 4, .to = 3, .weight = -3 },
    };

    const dist = try bellmanFord(alloc, 5, &edges, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, -1, 2, -2, 1 }, dist);
}

test "bellman-ford: disconnected nodes keep infinity" {
    const alloc = testing.allocator;
    const inf = std.math.maxInt(i64);
    const edges = [_]Edge{
        .{ .from = 0, .to = 1, .weight = 3 },
        .{ .from = 2, .to = 3, .weight = 4 },
    };

    const dist = try bellmanFord(alloc, 4, &edges, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 3, inf, inf }, dist);
}

test "bellman-ford: invalid start returns empty" {
    const alloc = testing.allocator;
    const edges = [_]Edge{};
    const dist = try bellmanFord(alloc, 3, &edges, 9);
    defer alloc.free(dist);
    try testing.expectEqual(@as(usize, 0), dist.len);
}

test "bellman-ford: invalid edge endpoints are ignored" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .from = 0, .to = 1, .weight = 2 },
        .{ .from = 0, .to = 9, .weight = 1 },
        .{ .from = 8, .to = 1, .weight = 1 },
    };

    const dist = try bellmanFord(alloc, 2, &edges, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 2 }, dist);
}

test "bellman-ford: detects negative cycle" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 1, .to = 2, .weight = -1 },
        .{ .from = 2, .to = 1, .weight = -1 },
    };
    try testing.expectError(error.NegativeCycle, bellmanFord(alloc, 3, &edges, 0));
}
