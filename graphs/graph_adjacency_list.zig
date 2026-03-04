//! Graph Adjacency List Data Structure - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/graph_adjacency_list.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const GraphAdjacencyList = struct {
    allocator: Allocator,
    directed: bool,
    adjacency: []std.ArrayListUnmanaged(usize),

    pub fn init(allocator: Allocator, vertex_count: usize, directed: bool) !GraphAdjacencyList {
        const adjacency = try allocator.alloc(std.ArrayListUnmanaged(usize), vertex_count);
        for (0..vertex_count) |i| adjacency[i] = .{};
        return .{
            .allocator = allocator,
            .directed = directed,
            .adjacency = adjacency,
        };
    }

    pub fn deinit(self: *GraphAdjacencyList) void {
        for (self.adjacency) |*neighbors| neighbors.deinit(self.allocator);
        self.allocator.free(self.adjacency);
    }

    pub fn vertexCount(self: *const GraphAdjacencyList) usize {
        return self.adjacency.len;
    }

    pub fn addEdge(self: *GraphAdjacencyList, u: usize, v: usize) !void {
        try self.validateVertex(u);
        try self.validateVertex(v);

        if (try self.containsEdge(u, v)) return error.EdgeExists;
        try self.adjacency[u].append(self.allocator, v);

        if (!self.directed) {
            if (try self.containsEdge(v, u)) return error.EdgeExists;
            try self.adjacency[v].append(self.allocator, u);
        }
    }

    pub fn removeEdge(self: *GraphAdjacencyList, u: usize, v: usize) !void {
        try self.validateVertex(u);
        try self.validateVertex(v);

        if (!removeFirstOccurrence(&self.adjacency[u], v)) return error.EdgeNotFound;
        if (!self.directed) {
            if (!removeFirstOccurrence(&self.adjacency[v], u)) return error.EdgeNotFound;
        }
    }

    pub fn containsEdge(self: *const GraphAdjacencyList, u: usize, v: usize) !bool {
        try self.validateVertex(u);
        try self.validateVertex(v);

        for (self.adjacency[u].items) |neighbor| {
            if (neighbor == v) return true;
        }
        return false;
    }

    fn validateVertex(self: *const GraphAdjacencyList, v: usize) !void {
        if (v >= self.adjacency.len) return error.InvalidVertex;
    }
};

fn removeFirstOccurrence(list: *std.ArrayListUnmanaged(usize), target: usize) bool {
    for (list.items, 0..) |value, index| {
        if (value == target) {
            _ = list.orderedRemove(index);
            return true;
        }
    }
    return false;
}

test "graph adjacency list: directed add/contains/remove" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyList.init(alloc, 4, true);
    defer graph.deinit();

    try graph.addEdge(0, 1);
    try graph.addEdge(1, 2);

    try testing.expect(try graph.containsEdge(0, 1));
    try testing.expect(!(try graph.containsEdge(1, 0)));

    try graph.removeEdge(0, 1);
    try testing.expect(!(try graph.containsEdge(0, 1)));
}

test "graph adjacency list: undirected mirrors edge" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyList.init(alloc, 3, false);
    defer graph.deinit();

    try graph.addEdge(0, 2);
    try testing.expect(try graph.containsEdge(0, 2));
    try testing.expect(try graph.containsEdge(2, 0));
}

test "graph adjacency list: duplicate and invalid edge handling" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyList.init(alloc, 2, true);
    defer graph.deinit();

    try graph.addEdge(0, 1);
    try testing.expectError(error.EdgeExists, graph.addEdge(0, 1));
    try testing.expectError(error.InvalidVertex, graph.addEdge(0, 3));
    try testing.expectError(error.EdgeNotFound, graph.removeEdge(1, 0));
}

test "graph adjacency list: extreme chain" {
    const alloc = testing.allocator;
    const n: usize = 400;
    var graph = try GraphAdjacencyList.init(alloc, n, false);
    defer graph.deinit();

    for (0..n - 1) |i| {
        try graph.addEdge(i, i + 1);
    }

    try testing.expectEqual(n, graph.vertexCount());
    try testing.expect(try graph.containsEdge(0, 1));
    try testing.expect(try graph.containsEdge(n - 1, n - 2));
}
