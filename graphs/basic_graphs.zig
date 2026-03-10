//! Basic Graph Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/basic_graphs.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const bfs_impl = @import("bfs.zig");
const dfs_impl = @import("dfs.zig");
const topo_impl = @import("kahn_topological_sort.zig");
const dijkstra_impl = @import("dijkstra.zig");

pub const Edge = struct {
    from: usize,
    to: usize,
};

pub const WeightedEdge = struct {
    from: usize,
    to: usize,
    weight: u64,
};

pub const WeightedNeighbor = dijkstra_impl.Edge;

/// Frees an owned adjacency list created by the helpers in this module.
pub fn freeAdjacencyList(allocator: Allocator, adj: [][]usize) void {
    for (adj) |row| allocator.free(row);
    allocator.free(adj);
}

/// Frees an owned weighted adjacency list created by this module.
pub fn freeWeightedAdjacencyList(allocator: Allocator, adj: [][]WeightedNeighbor) void {
    for (adj) |row| allocator.free(row);
    allocator.free(adj);
}

/// Builds a directed unweighted graph from edge pairs.
/// Invalid edges are rejected.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn initializeUnweightedDirectedGraph(
    allocator: Allocator,
    node_count: usize,
    edges: []const Edge,
) ![][]usize {
    return buildUnweightedGraph(allocator, node_count, edges, true);
}

/// Builds an undirected unweighted graph from edge pairs.
/// Invalid edges are rejected.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn initializeUnweightedUndirectedGraph(
    allocator: Allocator,
    node_count: usize,
    edges: []const Edge,
) ![][]usize {
    return buildUnweightedGraph(allocator, node_count, edges, false);
}

/// Builds an undirected weighted graph from weighted edge tuples.
/// Invalid edges are rejected.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn initializeWeightedUndirectedGraph(
    allocator: Allocator,
    node_count: usize,
    edges: []const WeightedEdge,
) ![][]WeightedNeighbor {
    var lists = try allocator.alloc(std.ArrayListUnmanaged(WeightedNeighbor), node_count);
    defer allocator.free(lists);
    for (0..node_count) |i| lists[i] = .{};
    defer for (lists) |*list| list.deinit(allocator);

    for (edges) |edge| {
        if (edge.from >= node_count or edge.to >= node_count) return error.InvalidNode;
        try lists[edge.from].append(allocator, .{ .to = edge.to, .weight = edge.weight });
        try lists[edge.to].append(allocator, .{ .to = edge.from, .weight = edge.weight });
    }

    const out = try allocator.alloc([]WeightedNeighbor, node_count);
    errdefer {
        for (out[0..]) |row| allocator.free(row);
        allocator.free(out);
    }
    for (0..node_count) |i| {
        out[i] = try lists[i].toOwnedSlice(allocator);
    }
    return out;
}

/// Runs BFS traversal order on an adjacency-list graph.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn bfsTraversal(allocator: Allocator, adj: []const []const usize, start: usize) ![]usize {
    return bfs_impl.bfs(allocator, adj, start);
}

/// Runs DFS traversal order on an adjacency-list graph.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn dfsTraversal(allocator: Allocator, adj: []const []const usize, start: usize) ![]usize {
    return dfs_impl.dfs(allocator, adj, start);
}

/// Computes Dijkstra distances on a weighted adjacency-list graph.
/// Time complexity: O(V² + E), Space complexity: O(V)
pub fn dijkstraTraversal(
    allocator: Allocator,
    adj: []const []const WeightedNeighbor,
    start: usize,
) ![]u64 {
    return dijkstra_impl.dijkstra(allocator, adj, start);
}

/// Computes a topological order using Kahn's algorithm.
/// Returns `null` when the graph contains a cycle.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn topologicalTraversal(allocator: Allocator, adj: []const []const usize) !?[]usize {
    return topo_impl.kahnTopologicalSort(allocator, adj);
}

fn buildUnweightedGraph(
    allocator: Allocator,
    node_count: usize,
    edges: []const Edge,
    directed: bool,
) ![][]usize {
    var lists = try allocator.alloc(std.ArrayListUnmanaged(usize), node_count);
    defer allocator.free(lists);
    for (0..node_count) |i| lists[i] = .{};
    defer for (lists) |*list| list.deinit(allocator);

    for (edges) |edge| {
        if (edge.from >= node_count or edge.to >= node_count) return error.InvalidNode;
        try lists[edge.from].append(allocator, edge.to);
        if (!directed) {
            try lists[edge.to].append(allocator, edge.from);
        }
    }

    const out = try allocator.alloc([]usize, node_count);
    errdefer {
        for (out[0..]) |row| allocator.free(row);
        allocator.free(out);
    }
    for (0..node_count) |i| {
        out[i] = try lists[i].toOwnedSlice(allocator);
    }
    return out;
}

test "basic graphs: initialize directed graph and traverse" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .from = 0, .to = 1 },
        .{ .from = 0, .to = 2 },
        .{ .from = 1, .to = 3 },
    };

    const adj = try initializeUnweightedDirectedGraph(alloc, 4, &edges);
    defer freeAdjacencyList(alloc, adj);

    const bfs_order = try bfsTraversal(alloc, adj, 0);
    defer alloc.free(bfs_order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, bfs_order);

    const dfs_order = try dfsTraversal(alloc, adj, 0);
    defer alloc.free(dfs_order);
    try testing.expectEqual(@as(usize, 4), dfs_order.len);
    try testing.expectEqual(@as(usize, 0), dfs_order[0]);
}

test "basic graphs: initialize undirected graph mirrors edges" {
    const alloc = testing.allocator;
    const edges = [_]Edge{
        .{ .from = 0, .to = 1 },
        .{ .from = 1, .to = 2 },
    };

    const adj = try initializeUnweightedUndirectedGraph(alloc, 3, &edges);
    defer freeAdjacencyList(alloc, adj);

    try testing.expectEqualSlices(usize, &[_]usize{1}, adj[0]);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 2 }, adj[1]);
    try testing.expectEqualSlices(usize, &[_]usize{1}, adj[2]);
}

test "basic graphs: weighted graph dijkstra and topo" {
    const alloc = testing.allocator;
    const weighted_edges = [_]WeightedEdge{
        .{ .from = 0, .to = 1, .weight = 4 },
        .{ .from = 0, .to = 2, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 7 },
    };

    const weighted_adj = try initializeWeightedUndirectedGraph(alloc, 4, &weighted_edges);
    defer freeWeightedAdjacencyList(alloc, weighted_adj);

    const dist = try dijkstraTraversal(alloc, weighted_adj, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 4, 1, 3 }, dist);

    const dag_edges = [_]Edge{
        .{ .from = 0, .to = 1 },
        .{ .from = 0, .to = 2 },
        .{ .from = 1, .to = 3 },
        .{ .from = 2, .to = 3 },
    };
    const dag = try initializeUnweightedDirectedGraph(alloc, 4, &dag_edges);
    defer freeAdjacencyList(alloc, dag);

    const topo = (try topologicalTraversal(alloc, dag)).?;
    defer alloc.free(topo);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, topo);
}

test "basic graphs: invalid edge is rejected" {
    const alloc = testing.allocator;
    const invalid = [_]Edge{
        .{ .from = 0, .to = 2 },
    };
    try testing.expectError(error.InvalidNode, initializeUnweightedDirectedGraph(alloc, 2, &invalid));

    const weighted_invalid = [_]WeightedEdge{
        .{ .from = 0, .to = 5, .weight = 1 },
    };
    try testing.expectError(error.InvalidNode, initializeWeightedUndirectedGraph(alloc, 2, &weighted_invalid));
}

test "basic graphs: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 256;
    const edges = try alloc.alloc(Edge, n - 1);
    defer alloc.free(edges);
    for (0..n - 1) |i| {
        edges[i] = .{ .from = i, .to = i + 1 };
    }

    const adj = try initializeUnweightedDirectedGraph(alloc, n, edges);
    defer freeAdjacencyList(alloc, adj);

    const bfs_order = try bfsTraversal(alloc, adj, 0);
    defer alloc.free(bfs_order);
    try testing.expectEqual(n, bfs_order.len);
    for (bfs_order, 0..) |value, i| {
        try testing.expectEqual(i, value);
    }
}
