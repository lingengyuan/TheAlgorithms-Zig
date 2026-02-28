//! Breadth-First Search (BFS) - Zig implementation
//! Uses an adjacency list represented as slices of slices.
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Performs BFS from `start` on an adjacency list graph.
/// Returns the order in which nodes were visited.
/// Caller owns the returned slice.
/// Invalid neighbor indices (>= adj.len) are silently skipped.
pub fn bfs(allocator: Allocator, adj: []const []const usize, start: usize) ![]usize {
    const n = adj.len;
    if (start >= n) return try allocator.alloc(usize, 0);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    // Ring buffer queue for O(1) dequeue
    var queue_buf = std.ArrayListUnmanaged(usize){};
    defer queue_buf.deinit(allocator);
    var queue_head: usize = 0;

    var result = std.ArrayListUnmanaged(usize){};
    defer result.deinit(allocator);

    visited[start] = true;
    try queue_buf.append(allocator, start);

    while (queue_head < queue_buf.items.len) {
        const current = queue_buf.items[queue_head];
        queue_head += 1;
        try result.append(allocator, current);

        for (adj[current]) |neighbor| {
            if (neighbor >= n) continue; // skip invalid neighbor
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                try queue_buf.append(allocator, neighbor);
            }
        }
    }

    const out = try allocator.alloc(usize, result.items.len);
    @memcpy(out, result.items);
    return out;
}

// ===== Tests =====

test "bfs: simple graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{ 0, 3, 4 }, // 1
        &[_]usize{0}, // 2
        &[_]usize{1}, // 3
        &[_]usize{1}, // 4
    };

    const order = try bfs(alloc, &adj, 0);
    defer alloc.free(order);

    try testing.expectEqual(@as(usize, 5), order.len);
    try testing.expectEqual(@as(usize, 0), order[0]);
    try testing.expectEqual(@as(usize, 1), order[1]);
    try testing.expectEqual(@as(usize, 2), order[2]);
    try testing.expectEqual(@as(usize, 3), order[3]);
    try testing.expectEqual(@as(usize, 4), order[4]);
}

test "bfs: disconnected graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{3},
        &[_]usize{2},
    };

    const order = try bfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 2), order.len);
}

test "bfs: single node" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{},
    };

    const order = try bfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{0}, order);
}

test "bfs: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};
    const order = try bfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 0), order.len);
}

test "bfs: invalid neighbor index is skipped" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 9 }, // 9 is out of bounds, should be skipped
        &[_]usize{0},
    };

    const order = try bfs(alloc, &adj, 0);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 2), order.len);
    try testing.expectEqual(@as(usize, 0), order[0]);
    try testing.expectEqual(@as(usize, 1), order[1]);
}
