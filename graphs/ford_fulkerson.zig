//! Ford-Fulkerson Max Flow - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/networking_flow/ford_fulkerson.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes maximum flow from `source` to `sink` using BFS-based augmenting paths
/// (Edmonds-Karp specialization of Ford-Fulkerson).
/// Input is an adjacency matrix of capacities.
/// Time complexity: O(V^5) worst-case for dense adjacency-matrix traversal,
/// Space complexity: O(V^2)
pub fn fordFulkersonMaxFlow(
    allocator: Allocator,
    capacity: []const []const i64,
    source: usize,
    sink: usize,
) !i64 {
    const n = capacity.len;
    if (source >= n or sink >= n) return error.InvalidNode;
    if (source == sink) return 0;

    for (capacity) |row| {
        if (row.len != n) return error.InvalidMatrix;
        for (row) |cap| {
            if (cap < 0) return error.NegativeCapacity;
        }
    }

    const residual = try allocator.alloc(i64, n * n);
    defer allocator.free(residual);
    for (capacity, 0..) |row, i| {
        @memcpy(residual[i * n .. (i + 1) * n], row);
    }

    const parent = try allocator.alloc(usize, n);
    defer allocator.free(parent);
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);

    const none = std.math.maxInt(usize);
    var max_flow: i64 = 0;

    while (true) {
        @memset(parent, none);
        @memset(visited, false);
        queue.clearRetainingCapacity();

        try queue.append(allocator, source);
        visited[source] = true;
        parent[source] = source;

        var head: usize = 0;
        search: while (head < queue.items.len) {
            const u = queue.items[head];
            head += 1;

            for (0..n) |v| {
                const cap = residual[u * n + v];
                if (!visited[v] and cap > 0) {
                    visited[v] = true;
                    parent[v] = u;
                    if (v == sink) break :search;
                    try queue.append(allocator, v);
                }
            }
        }

        if (!visited[sink]) break;

        var path_flow: i64 = std.math.maxInt(i64);
        var v = sink;
        while (v != source) {
            const u = parent[v];
            if (u == none) return error.InternalInvariantBroken;
            const cap = residual[u * n + v];
            if (cap < path_flow) path_flow = cap;
            v = u;
        }
        if (path_flow <= 0) return error.InternalInvariantBroken;

        const flow_sum = @addWithOverflow(max_flow, path_flow);
        if (flow_sum[1] != 0) return error.Overflow;
        max_flow = flow_sum[0];

        v = sink;
        while (v != source) {
            const u = parent[v];
            if (u == none) return error.InternalInvariantBroken;

            residual[u * n + v] -= path_flow;

            const rev_sum = @addWithOverflow(residual[v * n + u], path_flow);
            if (rev_sum[1] != 0) return error.Overflow;
            residual[v * n + u] = rev_sum[0];

            v = u;
        }
    }

    return max_flow;
}

test "ford fulkerson: classic sample graph" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 16, 13, 0, 0, 0 },
        &[_]i64{ 0, 0, 10, 12, 0, 0 },
        &[_]i64{ 0, 4, 0, 0, 14, 0 },
        &[_]i64{ 0, 0, 9, 0, 0, 20 },
        &[_]i64{ 0, 0, 0, 7, 0, 4 },
        &[_]i64{ 0, 0, 0, 0, 0, 0 },
    };

    try testing.expectEqual(@as(i64, 23), try fordFulkersonMaxFlow(alloc, &capacity, 0, 5));
}

test "ford fulkerson: unreachable sink returns zero" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 5, 0, 0 },
        &[_]i64{ 0, 0, 0, 0 },
        &[_]i64{ 0, 0, 0, 3 },
        &[_]i64{ 0, 0, 0, 0 },
    };

    try testing.expectEqual(@as(i64, 0), try fordFulkersonMaxFlow(alloc, &capacity, 0, 3));
}

test "ford fulkerson: source equals sink returns zero" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 0, 0 },
    };
    try testing.expectEqual(@as(i64, 0), try fordFulkersonMaxFlow(alloc, &capacity, 0, 0));
}

test "ford fulkerson: invalid node returns error" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 0, 0 },
    };

    try testing.expectError(error.InvalidNode, fordFulkersonMaxFlow(alloc, &capacity, 2, 1));
    try testing.expectError(error.InvalidNode, fordFulkersonMaxFlow(alloc, &capacity, 0, 2));
}

test "ford fulkerson: non-square matrix returns error" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 1, 2 },
        &[_]i64{ 0, 0 },
    };
    try testing.expectError(error.InvalidMatrix, fordFulkersonMaxFlow(alloc, &capacity, 0, 1));
}

test "ford fulkerson: negative capacity returns error" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, -1 },
        &[_]i64{ 0, 0 },
    };
    try testing.expectError(error.NegativeCapacity, fordFulkersonMaxFlow(alloc, &capacity, 0, 1));
}

test "ford fulkerson: overflow-prone total flow returns error" {
    const alloc = testing.allocator;
    const m = std.math.maxInt(i64);
    const capacity = [_][]const i64{
        &[_]i64{ 0, m, m, 0 },
        &[_]i64{ 0, 0, 0, m },
        &[_]i64{ 0, 0, 0, m },
        &[_]i64{ 0, 0, 0, 0 },
    };

    try testing.expectError(error.Overflow, fordFulkersonMaxFlow(alloc, &capacity, 0, 3));
}
