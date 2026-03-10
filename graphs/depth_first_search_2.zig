//! Depth First Search (Iterative Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/depth_first_search_2.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Runs iterative DFS from `start` over an adjacency-list graph.
/// Invalid neighbor indices are ignored.
/// Returns traversal order.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn depthFirstSearch(allocator: Allocator, graph: []const []const usize, start: usize) ![]usize {
    const n = graph.len;
    if (start >= n) return try allocator.alloc(usize, 0);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);
    visited[start] = true;
    try stack.append(allocator, start);

    var order = std.ArrayListUnmanaged(usize){};
    errdefer order.deinit(allocator);

    while (stack.items.len > 0) {
        const node = stack.items[stack.items.len - 1];
        _ = stack.pop();

        if (node >= n) continue;
        try order.append(allocator, node);

        // Push neighbors in reverse so lower index neighbors are visited first.
        var i = graph[node].len;
        while (i > 0) {
            i -= 1;
            const next = graph[node][i];
            if (next >= n or visited[next]) continue;
            visited[next] = true;
            try stack.append(allocator, next);
        }
    }

    return try order.toOwnedSlice(allocator);
}

test "depth first search 2: simple connected graph" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 3, 4 },
        &[_]usize{0},
        &[_]usize{1},
        &[_]usize{1},
    };

    const order = try depthFirstSearch(alloc, &graph, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 3, 4, 2 }, order);
}

test "depth first search 2: invalid start returns empty" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{&[_]usize{}};

    const order = try depthFirstSearch(alloc, &graph, 8);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 0), order.len);
}

test "depth first search 2: cycle graph does not loop" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{0},
    };

    const order = try depthFirstSearch(alloc, &graph, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, order);
}

test "depth first search 2: extreme deep chain" {
    const alloc = testing.allocator;
    const n: usize = 1024;

    const graph = try alloc.alloc([]const usize, n);
    defer alloc.free(graph);

    const edges = try alloc.alloc(usize, n - 1);
    defer alloc.free(edges);

    for (0..n - 1) |i| edges[i] = i + 1;
    for (0..n - 1) |i| graph[i] = edges[i .. i + 1];
    graph[n - 1] = &[_]usize{};

    const order = try depthFirstSearch(alloc, graph, 0);
    defer alloc.free(order);

    try testing.expectEqual(n, order.len);
    try testing.expectEqual(@as(usize, 0), order[0]);
    try testing.expectEqual(n - 1, order[n - 1]);
}

test "depth first search 2: shared successor is visited once" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{3},
        &[_]usize{3},
        &[_]usize{},
    };

    const order = try depthFirstSearch(alloc, &graph, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 3, 2 }, order);
}
