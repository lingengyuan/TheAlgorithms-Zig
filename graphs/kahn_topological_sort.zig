//! Kahn Topological Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/kahns_algorithm_topo.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Performs topological sorting of a directed graph using Kahn's algorithm.
/// Returns `null` when a cycle exists (no topological ordering).
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn kahnTopologicalSort(allocator: Allocator, adj: []const []const usize) !?[]usize {
    const n = adj.len;
    const indegree = try allocator.alloc(usize, n);
    defer allocator.free(indegree);
    @memset(indegree, 0);

    for (adj) |neighbors| {
        for (neighbors) |v| {
            if (v >= n) continue;
            const sum = @addWithOverflow(indegree[v], 1);
            if (sum[1] != 0) return error.Overflow;
            indegree[v] = sum[0];
        }
    }

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    var head: usize = 0;

    for (0..n) |v| {
        if (indegree[v] == 0) {
            try queue.append(allocator, v);
        }
    }

    var order = std.ArrayListUnmanaged(usize){};
    defer order.deinit(allocator);

    while (head < queue.items.len) {
        const u = queue.items[head];
        head += 1;
        try order.append(allocator, u);

        for (adj[u]) |v| {
            if (v >= n) continue;
            if (indegree[v] == 0) continue;
            indegree[v] -= 1;
            if (indegree[v] == 0) {
                try queue.append(allocator, v);
            }
        }
    }

    if (order.items.len != n) return null;
    return try order.toOwnedSlice(allocator);
}

test "kahn topo sort: python sample DAG" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{3}, // 1
        &[_]usize{3}, // 2
        &[_]usize{ 4, 5 }, // 3
        &[_]usize{}, // 4
        &[_]usize{}, // 5
    };

    const order = (try kahnTopologicalSort(alloc, &adj)).?;
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3, 4, 5 }, order);
}

test "kahn topo sort: cycle returns null" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{0},
    };

    const order = try kahnTopologicalSort(alloc, &adj);
    try testing.expect(order == null);
}

test "kahn topo sort: invalid neighbors are ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 100 },
        &[_]usize{},
        &[_]usize{},
    };

    const order = (try kahnTopologicalSort(alloc, &adj)).?;
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 3), order.len);
    try testing.expectEqual(@as(usize, 0), order[0]);
}

test "kahn topo sort: extreme long chain DAG" {
    const alloc = testing.allocator;
    const n: usize = 256;

    const mutable_adj = try alloc.alloc([]usize, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }

    for (0..n) |i| {
        if (i + 1 < n) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = i + 1;
        } else {
            mutable_adj[i] = try alloc.alloc(usize, 0);
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    const order = (try kahnTopologicalSort(alloc, adj)).?;
    defer alloc.free(order);

    try testing.expectEqual(n, order.len);
    for (order, 0..) |v, i| {
        try testing.expectEqual(i, v);
    }
}
