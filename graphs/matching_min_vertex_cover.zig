//! Matching-based Minimum Vertex Cover Approximation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/matching_min_vertex_cover.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Matching-based approximation for vertex cover.
/// Uses all directed edges from adjacency list; invalid neighbors are ignored.
/// Deterministically picks the lexicographically smallest remaining edge.
/// Returns selected vertices in ascending order.
/// Time complexity: O(V^3) worst-case, Space complexity: O(V^2)
pub fn matchingMinVertexCover(allocator: Allocator, graph: []const []const usize) ![]usize {
    const n = graph.len;
    if (n == 0) return try allocator.alloc(usize, 0);

    const edge_matrix = try allocator.alloc(bool, n * n);
    defer allocator.free(edge_matrix);
    @memset(edge_matrix, false);

    for (0..n) |u| {
        for (graph[u]) |v| {
            if (v >= n) continue;
            edge_matrix[u * n + v] = true;
        }
    }

    const chosen = try allocator.alloc(bool, n);
    defer allocator.free(chosen);
    @memset(chosen, false);

    while (true) {
        const edge_opt = findFirstEdge(edge_matrix, n);
        if (edge_opt == null) break;

        const u = edge_opt.?[0];
        const v = edge_opt.?[1];
        chosen[u] = true;
        chosen[v] = true;

        for (0..n) |k| {
            edge_matrix[u * n + k] = false;
            edge_matrix[k * n + u] = false;
            edge_matrix[v * n + k] = false;
            edge_matrix[k * n + v] = false;
        }
    }

    var count: usize = 0;
    for (chosen) |is_chosen| {
        if (is_chosen) count += 1;
    }

    const result = try allocator.alloc(usize, count);
    var idx: usize = 0;
    for (0..n) |node| {
        if (!chosen[node]) continue;
        result[idx] = node;
        idx += 1;
    }
    return result;
}

fn findFirstEdge(edge_matrix: []const bool, n: usize) ?[2]usize {
    for (0..n) |u| {
        for (0..n) |v| {
            if (edge_matrix[u * n + v]) return .{ u, v };
        }
    }
    return null;
}

test "matching min vertex cover: python sample" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 3 },
        &[_]usize{ 0, 3, 4 },
        &[_]usize{ 0, 1, 2 },
        &[_]usize{ 2, 3 },
    };

    const chosen = try matchingMinVertexCover(alloc, &graph);
    defer alloc.free(chosen);

    try testing.expectEqual(@as(usize, 4), chosen.len);
    try testing.expect(isVertexCover(&graph, chosen));
}

test "matching min vertex cover: no edges" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{},
        &[_]usize{},
        &[_]usize{},
    };

    const chosen = try matchingMinVertexCover(alloc, &graph);
    defer alloc.free(chosen);
    try testing.expectEqual(@as(usize, 0), chosen.len);
}

test "matching min vertex cover: ignores invalid neighbors" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{},
    };

    const chosen = try matchingMinVertexCover(alloc, &graph);
    defer alloc.free(chosen);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, chosen);
}

test "matching min vertex cover: extreme chain" {
    const alloc = testing.allocator;
    const n: usize = 201;

    const graph = try alloc.alloc([]const usize, n);
    defer alloc.free(graph);

    const edges = try alloc.alloc(usize, n - 1);
    defer alloc.free(edges);

    for (0..n - 1) |i| edges[i] = i + 1;
    for (0..n - 1) |i| graph[i] = edges[i .. i + 1];
    graph[n - 1] = &[_]usize{};

    const chosen = try matchingMinVertexCover(alloc, graph);
    defer alloc.free(chosen);

    try testing.expect(chosen.len > 0);
    try testing.expect(chosen.len <= n);
    try testing.expect(isVertexCover(graph, chosen));
}

fn isVertexCover(graph: []const []const usize, chosen: []const usize) bool {
    const n = graph.len;
    var selected = std.StaticBitSet(4096).initEmpty();
    if (n > 4096) return false;

    for (chosen) |v| {
        if (v >= n) return false;
        selected.set(v);
    }

    for (0..n) |u| {
        for (graph[u]) |v| {
            if (v >= n) continue;
            if (!selected.isSet(u) and !selected.isSet(v)) return false;
        }
    }
    return true;
}
