//! Kahn Longest Distance in DAG - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/kahns_algorithm_long.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes the longest path length in number of vertices using Kahn-style traversal.
/// For empty graph returns 0. Invalid neighbor indices are ignored.
/// If a cycle blocks processing, result follows Python behavior by returning
/// the maximum accumulated value among processed/default nodes.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn longestDistanceDag(allocator: Allocator, graph: []const []const usize) !usize {
    const n = graph.len;
    if (n == 0) return 0;

    const indegree = try allocator.alloc(usize, n);
    defer allocator.free(indegree);
    @memset(indegree, 0);

    for (graph) |neighbors| {
        for (neighbors) |v| {
            if (v < n) indegree[v] += 1;
        }
    }

    const long_dist = try allocator.alloc(usize, n);
    defer allocator.free(long_dist);
    @memset(long_dist, 1);

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    var head: usize = 0;

    for (0..n) |i| {
        if (indegree[i] == 0) try queue.append(allocator, i);
    }

    while (head < queue.items.len) {
        const u = queue.items[head];
        head += 1;

        for (graph[u]) |v| {
            if (v >= n) continue;

            if (long_dist[v] < long_dist[u] + 1) {
                long_dist[v] = long_dist[u] + 1;
            }

            indegree[v] -= 1;
            if (indegree[v] == 0) try queue.append(allocator, v);
        }
    }

    var best: usize = 0;
    for (long_dist) |value| {
        if (value > best) best = value;
    }
    return best;
}

test "kahn longest distance: python sample" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 2, 3, 4 },
        &[_]usize{ 2, 7 },
        &[_]usize{5},
        &[_]usize{ 5, 7 },
        &[_]usize{7},
        &[_]usize{6},
        &[_]usize{7},
        &[_]usize{},
    };

    const result = try longestDistanceDag(alloc, &graph);
    try testing.expectEqual(@as(usize, 5), result);
}

test "kahn longest distance: empty graph" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{};
    const result = try longestDistanceDag(alloc, &graph);
    try testing.expectEqual(@as(usize, 0), result);
}

test "kahn longest distance: ignores invalid neighbors" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{2},
        &[_]usize{},
    };
    const result = try longestDistanceDag(alloc, &graph);
    try testing.expectEqual(@as(usize, 3), result);
}

test "kahn longest distance: cycle returns default-compatible value" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{0},
    };

    const result = try longestDistanceDag(alloc, &graph);
    try testing.expectEqual(@as(usize, 1), result);
}

test "kahn longest distance: extreme linear dag" {
    const alloc = testing.allocator;
    const n: usize = 300;

    const graph = try alloc.alloc([]const usize, n);
    defer alloc.free(graph);

    const edges = try alloc.alloc(usize, n - 1);
    defer alloc.free(edges);

    for (0..n - 1) |i| edges[i] = i + 1;
    for (0..n - 1) |i| graph[i] = edges[i .. i + 1];
    graph[n - 1] = &[_]usize{};

    const result = try longestDistanceDag(alloc, graph);
    try testing.expectEqual(n, result);
}
