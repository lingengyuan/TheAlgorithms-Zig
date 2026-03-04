//! Random Graph Generator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/random_graph_generator.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const RandomGraph = struct {
    adjacency: []std.ArrayListUnmanaged(usize),

    pub fn deinit(self: RandomGraph, allocator: Allocator) void {
        for (self.adjacency) |*neighbors| neighbors.deinit(allocator);
        allocator.free(self.adjacency);
    }
};

/// Generates a random graph with `vertices` nodes.
/// Behavior matches Python reference semantics:
/// - `probability <= 0`: no edges.
/// - `probability >= 1`: complete *undirected* graph, even if directed=true.
/// - otherwise iterate i<j, add i->j with probability p; add reverse edge only when undirected.
/// Time complexity: O(V^2), Space complexity: O(V + E)
pub fn randomGraph(
    allocator: Allocator,
    vertices: usize,
    probability: f64,
    directed: bool,
    random: std.Random,
) !RandomGraph {
    const adjacency = try allocator.alloc(std.ArrayListUnmanaged(usize), vertices);
    errdefer allocator.free(adjacency);

    for (0..vertices) |i| adjacency[i] = .{};

    if (probability <= 0) {
        return .{ .adjacency = adjacency };
    }

    if (probability >= 1) {
        for (0..vertices) |i| {
            for (0..vertices) |j| {
                if (i == j) continue;
                try adjacency[i].append(allocator, j);
            }
        }
        return .{ .adjacency = adjacency };
    }

    for (0..vertices) |i| {
        for (i + 1..vertices) |j| {
            if (random.float(f64) < probability) {
                try adjacency[i].append(allocator, j);
                if (!directed) try adjacency[j].append(allocator, i);
            }
        }
    }

    return .{ .adjacency = adjacency };
}

fn edgeCount(graph: []const std.ArrayListUnmanaged(usize)) usize {
    var count: usize = 0;
    for (graph) |neighbors| count += neighbors.items.len;
    return count;
}

fn sameGraph(a: []const std.ArrayListUnmanaged(usize), b: []const std.ArrayListUnmanaged(usize)) bool {
    if (a.len != b.len) return false;
    for (a, b) |left, right| {
        if (!std.mem.eql(usize, left.items, right.items)) return false;
    }
    return true;
}

test "random graph generator: probability <= 0 gives empty graph" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);

    var graph = try randomGraph(alloc, 6, 0.0, false, prng.random());
    defer graph.deinit(alloc);

    try testing.expectEqual(@as(usize, 6), graph.adjacency.len);
    try testing.expectEqual(@as(usize, 0), edgeCount(graph.adjacency));
}

test "random graph generator: probability >= 1 yields complete undirected even when directed" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(13);

    var graph = try randomGraph(alloc, 4, 1.0, true, prng.random());
    defer graph.deinit(alloc);

    // Complete graph with 4 nodes has 4 * 3 directed entries.
    try testing.expectEqual(@as(usize, 12), edgeCount(graph.adjacency));

    for (0..4) |i| {
        try testing.expectEqual(@as(usize, 3), graph.adjacency[i].items.len);
    }
}

test "random graph generator: deterministic with same seed" {
    const alloc = testing.allocator;
    var prng_a = std.Random.DefaultPrng.init(12345);
    var prng_b = std.Random.DefaultPrng.init(12345);

    var graph_a = try randomGraph(alloc, 10, 0.35, false, prng_a.random());
    defer graph_a.deinit(alloc);

    var graph_b = try randomGraph(alloc, 10, 0.35, false, prng_b.random());
    defer graph_b.deinit(alloc);

    try testing.expect(sameGraph(graph_a.adjacency, graph_b.adjacency));
}

test "random graph generator: extreme sparse graph scale" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(999);

    var graph = try randomGraph(alloc, 256, 0.01, false, prng.random());
    defer graph.deinit(alloc);

    try testing.expectEqual(@as(usize, 256), graph.adjacency.len);
    for (graph.adjacency) |neighbors| {
        for (neighbors.items) |v| {
            try testing.expect(v < 256);
        }
    }
}
