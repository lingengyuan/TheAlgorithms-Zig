//! Connected Components Count - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/connected_components.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Counts connected components using iterative DFS over an adjacency list.
/// Graph is traversed following edges in `adj` as provided.
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn countConnectedComponents(allocator: Allocator, adj: []const []const usize) !usize {
    const n = adj.len;
    if (n == 0) return 0;

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);

    var components: usize = 0;

    for (0..n) |start| {
        if (visited[start]) continue;
        components += 1;
        visited[start] = true;
        try stack.append(allocator, start);

        while (stack.items.len > 0) {
            const cur = stack.pop().?;
            for (adj[cur]) |nb| {
                if (nb >= n) continue;
                if (!visited[nb]) {
                    visited[nb] = true;
                    try stack.append(allocator, nb);
                }
            }
        }
    }

    return components;
}

test "connected components: single component" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{ 0, 2 },
        &[_]usize{1},
    };
    try testing.expectEqual(@as(usize, 1), try countConnectedComponents(alloc, &adj));
}

test "connected components: multiple components" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{3},
        &[_]usize{2},
        &[_]usize{},
    };
    try testing.expectEqual(@as(usize, 3), try countConnectedComponents(alloc, &adj));
}

test "connected components: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 9 },
        &[_]usize{},
    };
    try testing.expectEqual(@as(usize, 1), try countConnectedComponents(alloc, &adj));
}

test "connected components: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};
    try testing.expectEqual(@as(usize, 0), try countConnectedComponents(alloc, &adj));
}
