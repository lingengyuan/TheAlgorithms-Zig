//! Even Tree (Maximum Removable Edges) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/even_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const TreeEdge = struct {
    u: usize,
    v: usize,
};

/// Returns the maximum number of removable edges so every resulting component
/// has an even number of nodes.
/// Nodes are 1-based and expected in range [1..node_count].
/// Input must represent a valid tree.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn maxRemovableEdgesEvenForest(
    allocator: Allocator,
    node_count: usize,
    edges: []const TreeEdge,
) !usize {
    if (node_count == 0) return error.InvalidTree;
    if (edges.len != node_count - 1) return error.InvalidTree;

    const adj = try allocator.alloc(std.ArrayListUnmanaged(usize), node_count + 1);
    defer {
        for (adj) |*list| list.deinit(allocator);
        allocator.free(adj);
    }
    for (adj) |*list| list.* = .{};

    for (edges) |edge| {
        if (edge.u == 0 or edge.v == 0 or edge.u > node_count or edge.v > node_count) {
            return error.InvalidNode;
        }
        if (edge.u == edge.v) return error.InvalidTree;
        try adj[edge.u].append(allocator, edge.v);
        try adj[edge.v].append(allocator, edge.u);
    }

    const visited = try allocator.alloc(bool, node_count + 1);
    defer allocator.free(visited);
    @memset(visited, false);

    var cuts: usize = 0;
    _ = try dfsEven(1, 0, adj, visited, &cuts);

    for (1..node_count + 1) |node| {
        if (!visited[node]) return error.InvalidTree;
    }

    return cuts;
}

fn dfsEven(
    node: usize,
    parent: usize,
    adj: []const std.ArrayListUnmanaged(usize),
    visited: []bool,
    cuts: *usize,
) !usize {
    if (visited[node]) return error.InvalidTree; // cycle
    visited[node] = true;

    var subtree_size: usize = 1;
    for (adj[node].items) |neighbor| {
        if (neighbor == parent) continue;
        const child_size = try dfsEven(neighbor, node, adj, visited, cuts);
        if (child_size % 2 == 0) {
            cuts.* += 1;
        } else {
            const sum = @addWithOverflow(subtree_size, child_size);
            if (sum[1] != 0) return error.Overflow;
            subtree_size = sum[0];
        }
    }

    return subtree_size;
}

test "even tree: python sample returns 2 cuts" {
    const alloc = testing.allocator;
    const edges = [_]TreeEdge{
        .{ .u = 2, .v = 1 },
        .{ .u = 3, .v = 1 },
        .{ .u = 4, .v = 3 },
        .{ .u = 5, .v = 2 },
        .{ .u = 6, .v = 1 },
        .{ .u = 7, .v = 2 },
        .{ .u = 8, .v = 6 },
        .{ .u = 9, .v = 8 },
        .{ .u = 10, .v = 8 },
    };
    try testing.expectEqual(@as(usize, 2), try maxRemovableEdgesEvenForest(alloc, 10, &edges));
}

test "even tree: single node tree" {
    const alloc = testing.allocator;
    const edges = [_]TreeEdge{};
    try testing.expectEqual(@as(usize, 0), try maxRemovableEdgesEvenForest(alloc, 1, &edges));
}

test "even tree: invalid inputs" {
    const alloc = testing.allocator;
    const bad_node = [_]TreeEdge{
        .{ .u = 1, .v = 3 },
    };
    try testing.expectError(error.InvalidNode, maxRemovableEdgesEvenForest(alloc, 2, &bad_node));

    const bad_cycle = [_]TreeEdge{
        .{ .u = 1, .v = 2 },
        .{ .u = 2, .v = 3 },
        .{ .u = 3, .v = 1 },
    };
    try testing.expectError(error.InvalidTree, maxRemovableEdgesEvenForest(alloc, 3, &bad_cycle));

    const disconnected = [_]TreeEdge{
        .{ .u = 1, .v = 2 },
        .{ .u = 3, .v = 4 },
        .{ .u = 4, .v = 5 },
    };
    try testing.expectError(error.InvalidTree, maxRemovableEdgesEvenForest(alloc, 5, &disconnected));
}

test "even tree: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 200;
    const edges = try alloc.alloc(TreeEdge, n - 1);
    defer alloc.free(edges);
    for (1..n) |i| {
        edges[i - 1] = .{ .u = i, .v = i + 1 };
    }

    try testing.expectEqual(@as(usize, 99), try maxRemovableEdgesEvenForest(alloc, n, edges));
}
