//! Greedy Minimum Vertex Cover Approximation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/greedy_min_vertex_cover.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Greedy approximation for vertex cover.
/// Behavior follows Python reference style:
/// - repeatedly pick highest current-degree remaining node,
/// - remove one occurrence of that node from each other remaining adjacency list.
/// Invalid neighbors are ignored.
/// Returns selected vertices in ascending order.
/// Time complexity: O(V^2 + V*E), Space complexity: O(V + E)
pub fn greedyMinVertexCover(allocator: Allocator, graph: []const []const usize) ![]usize {
    const n = graph.len;
    if (n == 0) return try allocator.alloc(usize, 0);

    const mutable_adj = try allocator.alloc(std.ArrayListUnmanaged(usize), n);
    defer {
        for (mutable_adj) |*list| list.deinit(allocator);
        allocator.free(mutable_adj);
    }

    for (0..n) |i| {
        mutable_adj[i] = .{};
        for (graph[i]) |neighbor| {
            if (neighbor < n) try mutable_adj[i].append(allocator, neighbor);
        }
    }

    const active = try allocator.alloc(bool, n);
    defer allocator.free(active);
    @memset(active, true);

    const chosen = try allocator.alloc(bool, n);
    defer allocator.free(chosen);
    @memset(chosen, false);

    while (true) {
        var best_degree: usize = 0;
        var best_node: ?usize = null;

        for (0..n) |node| {
            if (!active[node]) continue;
            const degree = mutable_adj[node].items.len;
            if (best_node == null or degree > best_degree or (degree == best_degree and node < best_node.?)) {
                best_node = node;
                best_degree = degree;
            }
        }

        const pick = best_node orelse break;
        if (best_degree == 0) break;

        active[pick] = false;
        chosen[pick] = true;

        for (0..n) |node| {
            if (!active[node]) continue;
            removeFirstOccurrence(&mutable_adj[node], pick);
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

fn removeFirstOccurrence(list: *std.ArrayListUnmanaged(usize), target: usize) void {
    for (list.items, 0..) |value, index| {
        if (value == target) {
            _ = list.orderedRemove(index);
            return;
        }
    }
}

test "greedy min vertex cover: python sample" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 3 },
        &[_]usize{ 0, 3, 4 },
        &[_]usize{ 0, 1, 2 },
        &[_]usize{ 2, 3 },
    };

    const chosen = try greedyMinVertexCover(alloc, &graph);
    defer alloc.free(chosen);

    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 4 }, chosen);
}

test "greedy min vertex cover: empty edges" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{},
        &[_]usize{},
    };

    const chosen = try greedyMinVertexCover(alloc, &graph);
    defer alloc.free(chosen);
    try testing.expectEqual(@as(usize, 0), chosen.len);
}

test "greedy min vertex cover: invalid neighbors ignored" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 9 },
        &[_]usize{},
    };

    const chosen = try greedyMinVertexCover(alloc, &graph);
    defer alloc.free(chosen);
    try testing.expectEqualSlices(usize, &[_]usize{0}, chosen);
}

test "greedy min vertex cover: extreme star graph" {
    const alloc = testing.allocator;
    const n: usize = 256;

    const graph = try alloc.alloc([]const usize, n);
    defer alloc.free(graph);

    const center_neighbors = try alloc.alloc(usize, n - 1);
    defer alloc.free(center_neighbors);

    for (1..n) |i| center_neighbors[i - 1] = i;
    graph[0] = center_neighbors;

    const leaves = try alloc.alloc(usize, n - 1);
    defer alloc.free(leaves);
    @memset(leaves, 0);
    for (1..n) |i| graph[i] = leaves[i - 1 .. i];

    const chosen = try greedyMinVertexCover(alloc, graph);
    defer alloc.free(chosen);

    try testing.expectEqual(@as(usize, 1), chosen.len);
    try testing.expectEqual(@as(usize, 0), chosen[0]);
}
