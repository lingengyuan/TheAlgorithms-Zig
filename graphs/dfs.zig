//! Depth-First Search (DFS) - Zig implementation (iterative with explicit stack)
//! Uses an adjacency list represented as slices of slices.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Performs iterative DFS from `start` on an adjacency list graph.
/// Returns the order in which nodes were visited.
/// Caller owns the returned slice.
pub fn dfs(allocator: Allocator, adj: []const []const usize, start: usize) ![]usize {
    const n = adj.len;
    if (start >= n) return try allocator.alloc(usize, 0);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);

    var result = std.ArrayListUnmanaged(usize){};
    defer result.deinit(allocator);

    try stack.append(allocator, start);

    while (stack.items.len > 0) {
        const current = stack.pop().?;
        if (visited[current]) continue;
        visited[current] = true;
        try result.append(allocator, current);

        // Push neighbors in reverse order so that lower-index neighbors are visited first
        const neighbors = adj[current];
        var i: usize = neighbors.len;
        while (i > 0) {
            i -= 1;
            const nb = neighbors[i];
            if (nb >= n) continue; // skip invalid neighbor
            if (!visited[nb]) {
                try stack.append(allocator, nb);
            }
        }
    }

    const out = try allocator.alloc(usize, result.items.len);
    @memcpy(out, result.items);
    return out;
}

// ===== Tests =====

test "dfs: simple graph" {
    // Graph:
    //   0 -- 1 -- 3
    //   |    |
    //   2    4
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{ 0, 3, 4 }, // 1
        &[_]usize{0}, // 2
        &[_]usize{1}, // 3
        &[_]usize{1}, // 4
    };

    const order = try dfs(alloc, &adj, 0);
    defer alloc.free(order);

    try testing.expectEqual(@as(usize, 5), order.len);
    // DFS from 0: goes deep first. 0 -> 1 -> 3 -> 4 -> 2
    try testing.expectEqual(@as(usize, 0), order[0]);
    try testing.expectEqual(@as(usize, 1), order[1]);
    try testing.expectEqual(@as(usize, 3), order[2]);
    try testing.expectEqual(@as(usize, 4), order[3]);
    try testing.expectEqual(@as(usize, 2), order[4]);
}

test "dfs: disconnected graph" {
    // 0 -- 1    2 -- 3
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1}, // 0
        &[_]usize{0}, // 1
        &[_]usize{3}, // 2
        &[_]usize{2}, // 3
    };

    const order = try dfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 2), order.len);
}

test "dfs: single node" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{},
    };

    const order = try dfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{0}, order);
}

test "dfs: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};
    const order = try dfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 0), order.len);
}

test "dfs: invalid neighbor index is skipped" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 9 }, // 9 is out of bounds, should be skipped
        &[_]usize{0},
    };

    const order = try dfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 2), order.len);
}

test "dfs: linear chain" {
    // 0 -> 1 -> 2 -> 3
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{3},
        &[_]usize{},
    };

    const order = try dfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, order);
}
