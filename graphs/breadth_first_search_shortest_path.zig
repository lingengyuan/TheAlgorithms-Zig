//! Breadth-First Search Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search_shortest_path.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Finds a shortest path from `source` to `target` in an unweighted graph.
/// Returns node path as a slice (inclusive of source and target).
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn bfsShortestPath(
    allocator: Allocator,
    adj: []const []const usize,
    source: usize,
    target: usize,
) ![]usize {
    const n = adj.len;
    if (source >= n or target >= n) return error.InvalidNode;

    if (source == target) {
        const out = try allocator.alloc(usize, 1);
        out[0] = source;
        return out;
    }

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    const parent = try allocator.alloc(usize, n);
    defer allocator.free(parent);
    const none = std.math.maxInt(usize);

    @memset(visited, false);
    @memset(parent, none);

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    var head: usize = 0;

    visited[source] = true;
    parent[source] = source;
    try queue.append(allocator, source);

    while (head < queue.items.len) {
        const u = queue.items[head];
        head += 1;
        if (u == target) break;

        for (adj[u]) |v| {
            if (v >= n) continue;
            if (!visited[v]) {
                visited[v] = true;
                parent[v] = u;
                try queue.append(allocator, v);
            }
        }
    }

    if (!visited[target]) return error.NoPath;

    var reversed = std.ArrayListUnmanaged(usize){};
    defer reversed.deinit(allocator);

    var cur = target;
    while (cur != source) {
        try reversed.append(allocator, cur);
        const p = parent[cur];
        if (p == none) return error.InternalInvariantBroken;
        cur = p;
    }
    try reversed.append(allocator, source);

    std.mem.reverse(usize, reversed.items);
    return try reversed.toOwnedSlice(allocator);
}

test "bfs shortest path: sample graph path" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2, 4 }, // 0
        &[_]usize{ 0, 3, 4 }, // 1
        &[_]usize{ 0, 5, 6 }, // 2
        &[_]usize{ 1, 4 }, // 3
        &[_]usize{ 0, 1, 3 }, // 4
        &[_]usize{2}, // 5
        &[_]usize{2}, // 6
    };

    const path = try bfsShortestPath(alloc, &adj, 6, 3);
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 6, 2, 0, 1, 3 }, path);
}

test "bfs shortest path: source equals target" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
    };

    const path = try bfsShortestPath(alloc, &adj, 1, 1);
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{1}, path);
}

test "bfs shortest path: no path and invalid node" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{},
    };

    try testing.expectError(error.NoPath, bfsShortestPath(alloc, &adj, 0, 2));
    try testing.expectError(error.InvalidNode, bfsShortestPath(alloc, &adj, 3, 0));
}

test "bfs shortest path: invalid neighbors are ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{0},
    };

    const path = try bfsShortestPath(alloc, &adj, 0, 1);
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, path);
}

test "bfs shortest path: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 256;

    const mutable_adj = try alloc.alloc([]usize, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }

    for (0..n) |i| {
        if (i == 0) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = 1;
        } else if (i + 1 == n) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = i - 1;
        } else {
            mutable_adj[i] = try alloc.alloc(usize, 2);
            mutable_adj[i][0] = i - 1;
            mutable_adj[i][1] = i + 1;
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    const path = try bfsShortestPath(alloc, adj, 0, n - 1);
    defer alloc.free(path);

    try testing.expectEqual(n, path.len);
    for (path, 0..) |v, i| try testing.expectEqual(i, v);
}
