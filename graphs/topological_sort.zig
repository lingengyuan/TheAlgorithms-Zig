//! Topological Sort (Kahn's Algorithm) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/topological_sort.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns one valid topological order for a directed acyclic graph.
/// Graph is represented as adjacency lists over vertex ids [0, n).
/// Invalid neighbor indices are ignored.
/// Returns `error.CycleDetected` if graph contains a cycle.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn topologicalSort(allocator: Allocator, adj: []const []const usize) ![]usize {
    const n = adj.len;
    if (n == 0) return try allocator.alloc(usize, 0);

    const indegree = try allocator.alloc(usize, n);
    defer allocator.free(indegree);
    @memset(indegree, 0);

    for (adj) |neighbors| {
        for (neighbors) |nb| {
            if (nb >= n) continue;
            indegree[nb] += 1;
        }
    }

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    var head: usize = 0;

    for (indegree, 0..) |deg, i| {
        if (deg == 0) try queue.append(allocator, i);
    }

    var order = std.ArrayListUnmanaged(usize){};
    defer order.deinit(allocator);

    while (head < queue.items.len) {
        const cur = queue.items[head];
        head += 1;
        try order.append(allocator, cur);

        for (adj[cur]) |nb| {
            if (nb >= n) continue;
            indegree[nb] -= 1;
            if (indegree[nb] == 0) {
                try queue.append(allocator, nb);
            }
        }
    }

    if (order.items.len != n) return error.CycleDetected;

    const out = try allocator.alloc(usize, n);
    @memcpy(out, order.items);
    return out;
}

test "topological sort: simple dag" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{3}, // 1
        &[_]usize{3}, // 2
        &[_]usize{}, // 3
    };

    const order = try topologicalSort(alloc, &adj);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, order);
}

test "topological sort: cycle returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{0},
    };
    try testing.expectError(error.CycleDetected, topologicalSort(alloc, &adj));
}

test "topological sort: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 9 },
        &[_]usize{},
    };

    const order = try topologicalSort(alloc, &adj);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, order);
}

test "topological sort: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};
    const order = try topologicalSort(alloc, &adj);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 0), order.len);
}
