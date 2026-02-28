//! Kruskal Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/kruskal.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Edge = struct {
    u: usize,
    v: usize,
    weight: i64,
};

fn lessThanEdge(_: void, a: Edge, b: Edge) bool {
    return a.weight < b.weight;
}

fn find(parent: []usize, x: usize) usize {
    var root = x;
    while (parent[root] != root) {
        root = parent[root];
    }

    var cur = x;
    while (parent[cur] != cur) {
        const next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    return root;
}

fn unite(parent: []usize, rank: []u8, a: usize, b: usize) bool {
    var ra = find(parent, a);
    var rb = find(parent, b);
    if (ra == rb) return false;

    if (rank[ra] < rank[rb]) {
        const t = ra;
        ra = rb;
        rb = t;
    }
    parent[rb] = ra;
    if (rank[ra] == rank[rb]) rank[ra] += 1;
    return true;
}

/// Computes MST total weight for an undirected weighted graph using Kruskal.
/// Invalid edge endpoints are ignored.
/// Returns `error.DisconnectedGraph` if MST cannot span all vertices.
/// Time complexity: O(E log E), Space complexity: O(V + E)
pub fn kruskalMstWeight(allocator: Allocator, vertex_count: usize, edges: []const Edge) !i64 {
    if (vertex_count == 0) return 0;

    var valid_edges = std.ArrayListUnmanaged(Edge){};
    defer valid_edges.deinit(allocator);
    for (edges) |edge| {
        if (edge.u >= vertex_count or edge.v >= vertex_count) continue;
        try valid_edges.append(allocator, edge);
    }

    std.sort.heap(Edge, valid_edges.items, {}, lessThanEdge);

    const parent = try allocator.alloc(usize, vertex_count);
    defer allocator.free(parent);
    const rank = try allocator.alloc(u8, vertex_count);
    defer allocator.free(rank);
    for (0..vertex_count) |i| parent[i] = i;
    @memset(rank, 0);

    var selected: usize = 0;
    var total: i64 = 0;

    for (valid_edges.items) |edge| {
        if (!unite(parent, rank, edge.u, edge.v)) continue;

        const sum = @addWithOverflow(total, edge.weight);
        if (sum[1] != 0) return error.Overflow;
        total = sum[0];

        selected += 1;
        if (selected == vertex_count - 1) break;
    }

    if (selected != vertex_count - 1) return error.DisconnectedGraph;
    return total;
}

test "kruskal: basic mst weight" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .u = 0, .v = 1, .weight = 10 },
        .{ .u = 0, .v = 2, .weight = 6 },
        .{ .u = 0, .v = 3, .weight = 5 },
        .{ .u = 1, .v = 3, .weight = 15 },
        .{ .u = 2, .v = 3, .weight = 4 },
    };
    try testing.expectEqual(@as(i64, 19), try kruskalMstWeight(alloc, 4, &edges));
}

test "kruskal: disconnected graph returns error" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .u = 0, .v = 1, .weight = 1 },
        .{ .u = 2, .v = 3, .weight = 1 },
    };
    try testing.expectError(error.DisconnectedGraph, kruskalMstWeight(alloc, 4, &edges));
}

test "kruskal: invalid edge endpoints are ignored" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .u = 0, .v = 1, .weight = 2 },
        .{ .u = 1, .v = 2, .weight = 3 },
        .{ .u = 0, .v = 9, .weight = 1 },
        .{ .u = 7, .v = 2, .weight = 1 },
    };
    try testing.expectEqual(@as(i64, 5), try kruskalMstWeight(alloc, 3, &edges));
}

test "kruskal: supports negative weights" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .u = 0, .v = 1, .weight = -2 },
        .{ .u = 1, .v = 2, .weight = 3 },
        .{ .u = 0, .v = 2, .weight = 4 },
    };
    try testing.expectEqual(@as(i64, 1), try kruskalMstWeight(alloc, 3, &edges));
}

test "kruskal: single vertex mst is zero" {
    const alloc = testing.allocator;
    const edges = [_]Edge{};
    try testing.expectEqual(@as(i64, 0), try kruskalMstWeight(alloc, 1, &edges));
}
