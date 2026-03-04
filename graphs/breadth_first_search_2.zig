//! Breadth First Search (Queue/Deque Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search_2.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Runs BFS from `start` over an adjacency-list graph.
/// Invalid neighbor indices are ignored.
/// Returns traversal order.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn breadthFirstSearch(allocator: Allocator, graph: []const []const usize, start: usize) ![]usize {
    const n = graph.len;
    if (start >= n) return try allocator.alloc(usize, 0);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    var head: usize = 0;

    var order = std.ArrayListUnmanaged(usize){};
    errdefer order.deinit(allocator);

    visited[start] = true;
    try queue.append(allocator, start);

    while (head < queue.items.len) {
        const node = queue.items[head];
        head += 1;
        try order.append(allocator, node);

        for (graph[node]) |next| {
            if (next >= n) continue;
            if (visited[next]) continue;
            visited[next] = true;
            try queue.append(allocator, next);
        }
    }

    return try order.toOwnedSlice(allocator);
}

test "breadth first search 2: python-style sample order" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 }, // A
        &[_]usize{ 0, 3, 4 },
        &[_]usize{ 0, 5 },
        &[_]usize{1},
        &[_]usize{ 1, 5 },
        &[_]usize{ 2, 4 },
    };

    const order = try breadthFirstSearch(alloc, &graph, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3, 4, 5 }, order);
}

test "breadth first search 2: invalid start returns empty" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{&[_]usize{}};

    const order = try breadthFirstSearch(alloc, &graph, 2);
    defer alloc.free(order);
    try testing.expectEqual(@as(usize, 0), order.len);
}

test "breadth first search 2: ignores invalid neighbors" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 100 },
        &[_]usize{2},
        &[_]usize{},
    };

    const order = try breadthFirstSearch(alloc, &graph, 0);
    defer alloc.free(order);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, order);
}

test "breadth first search 2: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 257;

    const graph = try alloc.alloc([]const usize, n);
    defer alloc.free(graph);

    const edges = try alloc.alloc(usize, n - 1);
    defer alloc.free(edges);

    for (0..n - 1) |i| edges[i] = i + 1;
    for (0..n - 1) |i| graph[i] = edges[i .. i + 1];
    graph[n - 1] = &[_]usize{};

    const order = try breadthFirstSearch(alloc, graph, 0);
    defer alloc.free(order);

    try testing.expectEqual(n, order.len);
    try testing.expectEqual(@as(usize, 0), order[0]);
    try testing.expectEqual(n - 1, order[n - 1]);
}
