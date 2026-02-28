//! Dijkstra Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Edge = struct {
    to: usize,
    weight: u64,
};

/// Computes single-source shortest path distances on a non-negative weighted graph.
/// Graph is adjacency-list based: `adj[u]` contains outgoing edges from node u.
/// Returns a distance slice where unreachable nodes are `std.math.maxInt(u64)`.
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V² + E), Space complexity: O(V)
pub fn dijkstra(allocator: Allocator, adj: []const []const Edge, start: usize) ![]u64 {
    const n = adj.len;
    if (start >= n) return try allocator.alloc(u64, 0);

    const dist = try allocator.alloc(u64, n);
    errdefer allocator.free(dist);
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);

    @memset(dist, std.math.maxInt(u64));
    @memset(visited, false);
    dist[start] = 0;

    var iter: usize = 0;
    while (iter < n) : (iter += 1) {
        var min_dist: u64 = std.math.maxInt(u64);
        var min_idx: ?usize = null;

        for (0..n) |i| {
            if (!visited[i] and dist[i] < min_dist) {
                min_dist = dist[i];
                min_idx = i;
            }
        }

        const u = min_idx orelse break;
        visited[u] = true;

        for (adj[u]) |edge| {
            if (edge.to >= n) continue;
            if (visited[edge.to]) continue;
            if (dist[u] == std.math.maxInt(u64)) continue;

            const sum = @addWithOverflow(dist[u], edge.weight);
            if (sum[1] != 0) continue; // skip overflowed path
            if (sum[0] < dist[edge.to]) {
                dist[edge.to] = sum[0];
            }
        }
    }

    return dist;
}

test "dijkstra: basic weighted graph" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 4 }, .{ .to = 2, .weight = 1 } }, // 0
        &[_]Edge{.{ .to = 3, .weight = 1 }}, // 1
        &[_]Edge{ .{ .to = 1, .weight = 2 }, .{ .to = 3, .weight = 5 } }, // 2
        &[_]Edge{}, // 3
    };

    const dist = try dijkstra(alloc, &adj, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 3, 1, 4 }, dist);
}

test "dijkstra: disconnected nodes keep infinity" {
    const alloc = testing.allocator;
    const inf = std.math.maxInt(u64);
    const adj = [_][]const Edge{
        &[_]Edge{.{ .to = 1, .weight = 2 }},
        &[_]Edge{},
        &[_]Edge{.{ .to = 3, .weight = 1 }},
        &[_]Edge{},
    };

    const dist = try dijkstra(alloc, &adj, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 2, inf, inf }, dist);
}

test "dijkstra: invalid start returns empty" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{},
        &[_]Edge{},
    };
    const dist = try dijkstra(alloc, &adj, 5);
    defer alloc.free(dist);
    try testing.expectEqual(@as(usize, 0), dist.len);
}

test "dijkstra: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 1 }, .{ .to = 9, .weight = 1 } },
        &[_]Edge{},
    };

    const dist = try dijkstra(alloc, &adj, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1 }, dist);
}

test "dijkstra: zero-weight edges" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 0 }, .{ .to = 2, .weight = 10 } },
        &[_]Edge{.{ .to = 2, .weight = 1 }},
        &[_]Edge{},
    };

    const dist = try dijkstra(alloc, &adj, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 1 }, dist);
}
