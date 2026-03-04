//! Boruvka Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/minimum_spanning_tree_boruvka.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const WeightedEdge = struct {
    u: usize,
    v: usize,
    weight: i64,
};

pub const BoruvkaResult = struct {
    total_weight: i64,
    edges: []WeightedEdge,

    pub fn deinit(self: BoruvkaResult, allocator: Allocator) void {
        allocator.free(self.edges);
    }
};

/// Computes MST for an undirected weighted graph using Boruvka's algorithm.
/// Input edges are treated as undirected single edges (do not need mirrored pairs).
/// Self-loops are ignored. Returns `error.DisconnectedGraph` when MST does not exist.
/// Time complexity: O(E log V), Space complexity: O(V + E)
pub fn boruvkaMst(
    allocator: Allocator,
    vertex_count: usize,
    input_edges: []const WeightedEdge,
) !BoruvkaResult {
    if (vertex_count == 0 or vertex_count == 1) {
        return .{
            .total_weight = 0,
            .edges = try allocator.alloc(WeightedEdge, 0),
        };
    }

    var edges = std.ArrayListUnmanaged(WeightedEdge){};
    defer edges.deinit(allocator);
    for (input_edges) |edge| {
        if (edge.u >= vertex_count or edge.v >= vertex_count) return error.InvalidNode;
        if (edge.u == edge.v) continue;
        try edges.append(allocator, edge);
    }
    if (edges.items.len == 0) return error.DisconnectedGraph;

    var uf = try UnionFind.init(allocator, vertex_count);
    defer uf.deinit(allocator);

    const cheapest = try allocator.alloc(?usize, vertex_count);
    defer allocator.free(cheapest);

    var mst_edges = std.ArrayListUnmanaged(WeightedEdge){};
    errdefer mst_edges.deinit(allocator);

    var components = vertex_count;
    var total: i64 = 0;

    while (components > 1) {
        @memset(cheapest, null);

        for (edges.items, 0..) |edge, edge_idx| {
            const set_u = uf.find(edge.u);
            const set_v = uf.find(edge.v);
            if (set_u == set_v) continue;

            updateCheapest(cheapest, edges.items, set_u, edge_idx);
            updateCheapest(cheapest, edges.items, set_v, edge_idx);
        }

        var added_any = false;

        for (0..vertex_count) |set_id| {
            const edge_idx = cheapest[set_id] orelse continue;
            const edge = edges.items[edge_idx];
            if (!uf.unionSets(edge.u, edge.v)) continue;

            try mst_edges.append(allocator, edge);
            const sum = @addWithOverflow(total, edge.weight);
            if (sum[1] != 0) return error.Overflow;
            total = sum[0];
            components -= 1;
            added_any = true;
        }

        if (!added_any) return error.DisconnectedGraph;
    }

    return .{
        .total_weight = total,
        .edges = try mst_edges.toOwnedSlice(allocator),
    };
}

fn updateCheapest(cheapest: []?usize, edges: []const WeightedEdge, set_id: usize, candidate_idx: usize) void {
    if (cheapest[set_id] == null) {
        cheapest[set_id] = candidate_idx;
        return;
    }

    const existing_idx = cheapest[set_id].?;
    const existing = edges[existing_idx];
    const candidate = edges[candidate_idx];
    if (candidate.weight < existing.weight or (candidate.weight == existing.weight and candidate_idx < existing_idx)) {
        cheapest[set_id] = candidate_idx;
    }
}

const UnionFind = struct {
    parent: []usize,
    rank: []u8,
    allocator: Allocator,

    fn init(allocator: Allocator, n: usize) !UnionFind {
        const parent = try allocator.alloc(usize, n);
        errdefer allocator.free(parent);
        const rank = try allocator.alloc(u8, n);
        errdefer allocator.free(rank);

        for (0..n) |i| {
            parent[i] = i;
            rank[i] = 0;
        }
        return .{
            .parent = parent,
            .rank = rank,
            .allocator = allocator,
        };
    }

    fn deinit(self: *UnionFind, allocator: Allocator) void {
        allocator.free(self.parent);
        allocator.free(self.rank);
    }

    fn find(self: *UnionFind, x: usize) usize {
        if (self.parent[x] != x) {
            self.parent[x] = self.find(self.parent[x]);
        }
        return self.parent[x];
    }

    fn unionSets(self: *UnionFind, a: usize, b: usize) bool {
        var root_a = self.find(a);
        var root_b = self.find(b);
        if (root_a == root_b) return false;

        if (self.rank[root_a] < self.rank[root_b]) {
            std.mem.swap(usize, &root_a, &root_b);
        }
        self.parent[root_b] = root_a;
        if (self.rank[root_a] == self.rank[root_b]) {
            self.rank[root_a] += 1;
        }
        return true;
    }
};

test "boruvka mst: python sample-style graph" {
    const alloc = testing.allocator;
    const edges = [_]WeightedEdge{
        .{ .u = 0, .v = 1, .weight = 1 },
        .{ .u = 0, .v = 2, .weight = 1 },
        .{ .u = 2, .v = 3, .weight = 1 },
    };

    var result = try boruvkaMst(alloc, 4, &edges);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(i64, 3), result.total_weight);
    try testing.expectEqual(@as(usize, 3), result.edges.len);
}

test "boruvka mst: duplicate oriented edges still yields valid mst" {
    const alloc = testing.allocator;
    const edges = [_]WeightedEdge{
        .{ .u = 0, .v = 1, .weight = 2 },
        .{ .u = 1, .v = 0, .weight = 2 },
        .{ .u = 1, .v = 2, .weight = 1 },
        .{ .u = 2, .v = 3, .weight = 3 },
        .{ .u = 0, .v = 3, .weight = 10 },
    };

    var result = try boruvkaMst(alloc, 4, &edges);
    defer result.deinit(alloc);
    try testing.expectEqual(@as(i64, 6), result.total_weight);
    try testing.expectEqual(@as(usize, 3), result.edges.len);
}

test "boruvka mst: disconnected graph returns error" {
    const alloc = testing.allocator;
    const edges = [_]WeightedEdge{
        .{ .u = 0, .v = 1, .weight = 1 },
        .{ .u = 2, .v = 3, .weight = 1 },
    };
    try testing.expectError(error.DisconnectedGraph, boruvkaMst(alloc, 4, &edges));
}

test "boruvka mst: invalid node returns error and self-loop ignored" {
    const alloc = testing.allocator;
    const invalid = [_]WeightedEdge{
        .{ .u = 0, .v = 2, .weight = 1 },
    };
    try testing.expectError(error.InvalidNode, boruvkaMst(alloc, 2, &invalid));

    const loop_only = [_]WeightedEdge{
        .{ .u = 0, .v = 0, .weight = 5 },
    };
    try testing.expectError(error.DisconnectedGraph, boruvkaMst(alloc, 2, &loop_only));
}

test "boruvka mst: extreme long chain graph" {
    const alloc = testing.allocator;
    const n: usize = 129;
    const edges = try alloc.alloc(WeightedEdge, n - 1);
    defer alloc.free(edges);

    for (0..n - 1) |i| {
        edges[i] = .{
            .u = i,
            .v = i + 1,
            .weight = 1,
        };
    }

    var result = try boruvkaMst(alloc, n, edges);
    defer result.deinit(alloc);
    try testing.expectEqual(@as(i64, @intCast(n - 1)), result.total_weight);
    try testing.expectEqual(n - 1, result.edges.len);
}
