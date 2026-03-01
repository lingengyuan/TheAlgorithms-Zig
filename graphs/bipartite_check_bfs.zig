//! Bipartite Graph Check (BFS) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/check_bipatrite.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns true if the graph is bipartite using BFS 2-coloring.
/// Graph is adjacency-list based; invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn isBipartiteBfs(allocator: Allocator, adj: []const []const usize) !bool {
    const n = adj.len;
    if (n == 0) return true;

    const color = try allocator.alloc(i8, n);
    defer allocator.free(color);
    @memset(color, -1);

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);

    for (0..n) |start| {
        if (color[start] != -1) continue;

        color[start] = 0;
        queue.clearRetainingCapacity();
        try queue.append(allocator, start);
        var head: usize = 0;

        while (head < queue.items.len) {
            const cur = queue.items[head];
            head += 1;

            for (adj[cur]) |nb| {
                if (nb >= n) continue;

                if (color[nb] == -1) {
                    color[nb] = 1 - color[cur];
                    try queue.append(allocator, nb);
                } else if (color[nb] == color[cur]) {
                    return false;
                }
            }
        }
    }

    return true;
}

test "bipartite bfs: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};
    try testing.expect(try isBipartiteBfs(alloc, &adj));
}

test "bipartite bfs: even cycle is bipartite" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 2 },
    };
    try testing.expect(try isBipartiteBfs(alloc, &adj));
}

test "bipartite bfs: odd cycle is not bipartite" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
    };
    try testing.expect(!(try isBipartiteBfs(alloc, &adj)));
}

test "bipartite bfs: disconnected graph with odd component is not bipartite" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{ 3, 4 },
        &[_]usize{ 2, 4 },
        &[_]usize{ 2, 3 },
    };
    try testing.expect(!(try isBipartiteBfs(alloc, &adj)));
}

test "bipartite bfs: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{0},
    };
    try testing.expect(try isBipartiteBfs(alloc, &adj));
}

test "bipartite bfs: self loop is not bipartite" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{0},
    };
    try testing.expect(!(try isBipartiteBfs(alloc, &adj)));
}

test "bipartite bfs: single node is bipartite" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{},
    };
    try testing.expect(try isBipartiteBfs(alloc, &adj));
}
