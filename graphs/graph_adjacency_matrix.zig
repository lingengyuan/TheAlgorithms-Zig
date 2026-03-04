//! Graph Adjacency Matrix Data Structure - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/graph_adjacency_matrix.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const GraphAdjacencyMatrix = struct {
    allocator: Allocator,
    directed: bool,
    vertex_count: usize,
    matrix: []bool,

    pub fn init(allocator: Allocator, vertex_count: usize, directed: bool) !GraphAdjacencyMatrix {
        const matrix = try allocator.alloc(bool, vertex_count * vertex_count);
        @memset(matrix, false);
        return .{
            .allocator = allocator,
            .directed = directed,
            .vertex_count = vertex_count,
            .matrix = matrix,
        };
    }

    pub fn deinit(self: *GraphAdjacencyMatrix) void {
        self.allocator.free(self.matrix);
    }

    pub fn containsEdge(self: *const GraphAdjacencyMatrix, u: usize, v: usize) !bool {
        try self.validateVertex(u);
        try self.validateVertex(v);
        return self.matrix[self.index(u, v)];
    }

    pub fn addEdge(self: *GraphAdjacencyMatrix, u: usize, v: usize) !void {
        try self.validateVertex(u);
        try self.validateVertex(v);

        const idx_uv = self.index(u, v);
        if (self.matrix[idx_uv]) return error.EdgeExists;
        self.matrix[idx_uv] = true;

        if (!self.directed) {
            self.matrix[self.index(v, u)] = true;
        }
    }

    pub fn removeEdge(self: *GraphAdjacencyMatrix, u: usize, v: usize) !void {
        try self.validateVertex(u);
        try self.validateVertex(v);

        const idx_uv = self.index(u, v);
        if (!self.matrix[idx_uv]) return error.EdgeNotFound;
        self.matrix[idx_uv] = false;

        if (!self.directed) {
            self.matrix[self.index(v, u)] = false;
        }
    }

    pub fn addVertex(self: *GraphAdjacencyMatrix) !void {
        const old_n = self.vertex_count;
        const new_n = old_n + 1;

        const new_matrix = try self.allocator.alloc(bool, new_n * new_n);
        @memset(new_matrix, false);

        for (0..old_n) |i| {
            for (0..old_n) |j| {
                new_matrix[i * new_n + j] = self.matrix[i * old_n + j];
            }
        }

        self.allocator.free(self.matrix);
        self.matrix = new_matrix;
        self.vertex_count = new_n;
    }

    pub fn removeVertex(self: *GraphAdjacencyMatrix, vertex: usize) !void {
        try self.validateVertex(vertex);

        const old_n = self.vertex_count;
        const new_n = old_n - 1;

        if (new_n == 0) {
            self.allocator.free(self.matrix);
            self.matrix = try self.allocator.alloc(bool, 0);
            self.vertex_count = 0;
            return;
        }

        const new_matrix = try self.allocator.alloc(bool, new_n * new_n);
        @memset(new_matrix, false);

        var ni: usize = 0;
        for (0..old_n) |i| {
            if (i == vertex) continue;
            var nj: usize = 0;
            for (0..old_n) |j| {
                if (j == vertex) continue;
                new_matrix[ni * new_n + nj] = self.matrix[i * old_n + j];
                nj += 1;
            }
            ni += 1;
        }

        self.allocator.free(self.matrix);
        self.matrix = new_matrix;
        self.vertex_count = new_n;
    }

    fn validateVertex(self: *const GraphAdjacencyMatrix, vertex: usize) !void {
        if (vertex >= self.vertex_count) return error.InvalidVertex;
    }

    fn index(self: *const GraphAdjacencyMatrix, u: usize, v: usize) usize {
        return u * self.vertex_count + v;
    }
};

test "graph adjacency matrix: directed edge operations" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyMatrix.init(alloc, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1);
    try testing.expect(try graph.containsEdge(0, 1));
    try testing.expect(!(try graph.containsEdge(1, 0)));

    try graph.removeEdge(0, 1);
    try testing.expect(!(try graph.containsEdge(0, 1)));
}

test "graph adjacency matrix: undirected mirror behavior" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyMatrix.init(alloc, 3, false);
    defer graph.deinit();

    try graph.addEdge(1, 2);
    try testing.expect(try graph.containsEdge(1, 2));
    try testing.expect(try graph.containsEdge(2, 1));
}

test "graph adjacency matrix: add/remove vertex keeps old edges" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyMatrix.init(alloc, 2, true);
    defer graph.deinit();

    try graph.addEdge(0, 1);
    try graph.addVertex();
    try testing.expectEqual(@as(usize, 3), graph.vertex_count);
    try testing.expect(try graph.containsEdge(0, 1));

    try graph.removeVertex(1);
    try testing.expectEqual(@as(usize, 2), graph.vertex_count);
}

test "graph adjacency matrix: invalid and duplicate handling" {
    const alloc = testing.allocator;
    var graph = try GraphAdjacencyMatrix.init(alloc, 2, true);
    defer graph.deinit();

    try graph.addEdge(0, 1);
    try testing.expectError(error.EdgeExists, graph.addEdge(0, 1));
    try testing.expectError(error.InvalidVertex, graph.addEdge(0, 2));
    try testing.expectError(error.EdgeNotFound, graph.removeEdge(1, 0));
}

test "graph adjacency matrix: extreme chain" {
    const alloc = testing.allocator;
    const n: usize = 128;
    var graph = try GraphAdjacencyMatrix.init(alloc, n, false);
    defer graph.deinit();

    for (0..n - 1) |i| {
        try graph.addEdge(i, i + 1);
    }

    try testing.expect(try graph.containsEdge(0, 1));
    try testing.expect(try graph.containsEdge(n - 1, n - 2));
}
