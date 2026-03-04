//! Deep Clone Graph - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/deep_clone_graph.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const GraphNode = struct {
    value: i64,
    neighbors: []const usize,
};

/// Deep clones a graph represented as an array of nodes with neighbor index lists.
/// Returns an owned clone where each neighbor list is independently allocated.
/// Invalid neighbor indices return `error.InvalidNeighbor`.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn deepCloneGraph(allocator: Allocator, graph: []const GraphNode) ![]GraphNode {
    const out = try allocator.alloc(GraphNode, graph.len);
    for (0..out.len) |i| {
        out[i] = .{
            .value = 0,
            .neighbors = &[_]usize{},
        };
    }
    errdefer {
        for (0..graph.len) |i| {
            if (i < out.len and out[i].neighbors.len > 0) allocator.free(out[i].neighbors);
        }
        allocator.free(out);
    }

    for (graph, 0..) |node, i| {
        const neigh = try allocator.alloc(usize, node.neighbors.len);
        errdefer allocator.free(neigh);
        for (node.neighbors, 0..) |v, j| {
            if (v >= graph.len) return error.InvalidNeighbor;
            neigh[j] = v;
        }
        out[i] = .{
            .value = node.value,
            .neighbors = neigh,
        };
    }

    return out;
}

pub fn freeClonedGraph(allocator: Allocator, graph: []GraphNode) void {
    for (graph) |node| allocator.free(node.neighbors);
    allocator.free(graph);
}

test "deep clone graph: empty graph" {
    const alloc = testing.allocator;
    const g = [_]GraphNode{};
    const clone = try deepCloneGraph(alloc, &g);
    defer freeClonedGraph(alloc, clone);
    try testing.expectEqual(@as(usize, 0), clone.len);
}

test "deep clone graph: single node and two-node edge" {
    const alloc = testing.allocator;
    const g1 = [_]GraphNode{
        .{ .value = 1, .neighbors = &[_]usize{} },
    };
    const c1 = try deepCloneGraph(alloc, &g1);
    defer freeClonedGraph(alloc, c1);
    try testing.expectEqual(@as(i64, 1), c1[0].value);
    try testing.expectEqual(@as(usize, 0), c1[0].neighbors.len);

    const g2 = [_]GraphNode{
        .{ .value = 1, .neighbors = &[_]usize{1} },
        .{ .value = 2, .neighbors = &[_]usize{0} },
    };
    const c2 = try deepCloneGraph(alloc, &g2);
    defer freeClonedGraph(alloc, c2);
    try testing.expectEqual(@as(i64, 1), c2[0].value);
    try testing.expectEqual(@as(i64, 2), c2[1].value);
    try testing.expectEqualSlices(usize, &[_]usize{1}, c2[0].neighbors);
    try testing.expectEqualSlices(usize, &[_]usize{0}, c2[1].neighbors);

    // ensure deep copy of neighbor slice memory
    try testing.expect(@intFromPtr(c2[0].neighbors.ptr) != @intFromPtr(g2[0].neighbors.ptr));
}

test "deep clone graph: invalid neighbor index returns error" {
    const alloc = testing.allocator;
    const bad = [_]GraphNode{
        .{ .value = 1, .neighbors = &[_]usize{2} },
        .{ .value = 2, .neighbors = &[_]usize{} },
    };
    try testing.expectError(error.InvalidNeighbor, deepCloneGraph(alloc, &bad));
}

test "deep clone graph: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 400;

    const g = try alloc.alloc(GraphNode, n);
    defer {
        for (g) |node| alloc.free(node.neighbors);
        alloc.free(g);
    }

    for (0..n) |i| {
        if (i + 1 < n) {
            const neigh = try alloc.alloc(usize, 1);
            neigh[0] = i + 1;
            g[i] = .{ .value = @as(i64, @intCast(i)), .neighbors = neigh };
        } else {
            g[i] = .{ .value = @as(i64, @intCast(i)), .neighbors = try alloc.alloc(usize, 0) };
        }
    }

    const clone = try deepCloneGraph(alloc, g);
    defer freeClonedGraph(alloc, clone);

    try testing.expectEqual(n, clone.len);
    for (0..n) |i| {
        try testing.expectEqual(g[i].value, clone[i].value);
        try testing.expectEqual(g[i].neighbors.len, clone[i].neighbors.len);
        if (g[i].neighbors.len == 1) {
            try testing.expectEqual(g[i].neighbors[0], clone[i].neighbors[0]);
        }
    }
}
