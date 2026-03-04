//! Topological Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/topological_sort.py

const std = @import("std");
const testing = std.testing;

pub const TopologicalSortError = error{
    UnknownStartVertex,
    InvalidEdge,
};

/// Topological sort using iterative DFS post-order.
/// Returns vertices in post-order finish sequence, matching Python reference shape.
/// Caller owns returned slice.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn topologicalSort(
    allocator: std.mem.Allocator,
    start: usize,
    adjacency: []const []const usize,
) (TopologicalSortError || std.mem.Allocator.Error)![]usize {
    if (start >= adjacency.len) return error.UnknownStartVertex;

    const visited = try allocator.alloc(bool, adjacency.len);
    defer allocator.free(visited);
    @memset(visited, false);

    var order = std.ArrayListUnmanaged(usize){};
    defer order.deinit(allocator);

    const Frame = struct { node: usize, next_edge: usize };

    var stack = std.ArrayListUnmanaged(Frame){};
    defer stack.deinit(allocator);

    const startDfs = struct {
        fn run(
            alloc: std.mem.Allocator,
            adj: []const []const usize,
            vis: []bool,
            out: *std.ArrayListUnmanaged(usize),
            st: *std.ArrayListUnmanaged(Frame),
            root: usize,
        ) !void {
            try st.append(alloc, .{ .node = root, .next_edge = 0 });
            vis[root] = true;

            while (st.items.len > 0) {
                var top = &st.items[st.items.len - 1];
                if (top.next_edge < adj[top.node].len) {
                    const next = adj[top.node][top.next_edge];
                    top.next_edge += 1;
                    if (next >= adj.len) return error.InvalidEdge;
                    if (!vis[next]) {
                        vis[next] = true;
                        try st.append(alloc, .{ .node = next, .next_edge = 0 });
                    }
                } else {
                    try out.append(alloc, top.node);
                    _ = st.pop();
                }
            }
        }
    }.run;

    try startDfs(allocator, adjacency, visited, &order, &stack, start);
    for (0..adjacency.len) |v| {
        if (!visited[v]) try startDfs(allocator, adjacency, visited, &order, &stack, v);
    }

    return try order.toOwnedSlice(allocator);
}

test "topological sort: python sample graph output shape" {
    const alloc = testing.allocator;
    const g = [_][]const usize{
        &[_]usize{ 2, 1 }, // a -> c, b
        &[_]usize{ 3, 4 }, // b -> d, e
        &[_]usize{}, // c
        &[_]usize{}, // d
        &[_]usize{}, // e
    };

    const order = try topologicalSort(alloc, 0, &g);
    defer alloc.free(order);

    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 4, 1, 0 }, order);
}

test "topological sort: disconnected graph and invalid input" {
    const alloc = testing.allocator;
    const g = [_][]const usize{
        &[_]usize{1},
        &[_]usize{},
        &[_]usize{3},
        &[_]usize{},
    };
    const order = try topologicalSort(alloc, 2, &g);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 4), order.len);

    try testing.expectError(error.UnknownStartVertex, topologicalSort(alloc, 9, &g));

    const bad = [_][]const usize{
        &[_]usize{1},
        &[_]usize{5}, // invalid
    };
    try testing.expectError(error.InvalidEdge, topologicalSort(alloc, 0, &bad));
}

test "topological sort: extreme chain graph" {
    const alloc = testing.allocator;
    const n: usize = 100_000;
    const edges_storage = try alloc.alloc([]usize, n);
    defer {
        for (edges_storage) |e| alloc.free(e);
        alloc.free(edges_storage);
    }
    const adjacency = try alloc.alloc([]const usize, n);
    defer alloc.free(adjacency);

    for (0..n) |i| {
        if (i + 1 < n) {
            const e = try alloc.alloc(usize, 1);
            e[0] = i + 1;
            edges_storage[i] = e;
            adjacency[i] = e;
        } else {
            edges_storage[i] = try alloc.alloc(usize, 0);
            adjacency[i] = edges_storage[i];
        }
    }

    const order = try topologicalSort(alloc, 0, adjacency);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, n), order.len);
    // post-order of chain n-1..0
    try testing.expectEqual(@as(usize, n - 1), order[0]);
    try testing.expectEqual(@as(usize, 0), order[order.len - 1]);
}
