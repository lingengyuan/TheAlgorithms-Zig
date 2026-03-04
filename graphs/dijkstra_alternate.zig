//! Dijkstra (Alternate Matrix Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra_alternate.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes shortest distances from `source` on a non-negative weighted adjacency matrix.
/// Matrix entry `<= 0` means no edge (except diagonal 0).
/// Returns empty slice when `source` is out of range.
/// Time complexity: O(V^2), Space complexity: O(V)
pub fn dijkstraAlternate(allocator: Allocator, graph: []const []const i64, source: usize) ![]i64 {
    const n = graph.len;
    if (source >= n) return try allocator.alloc(i64, 0);

    for (graph) |row| {
        if (row.len != n) return error.InvalidGraph;
    }

    const inf: i64 = 10_000_000;
    const dist = try allocator.alloc(i64, n);
    errdefer allocator.free(dist);
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);

    @memset(dist, inf);
    @memset(visited, false);
    dist[source] = 0;

    for (0..n) |_| {
        var min_distance = inf;
        var u_opt: ?usize = null;

        for (0..n) |v| {
            if (!visited[v] and dist[v] < min_distance) {
                min_distance = dist[v];
                u_opt = v;
            }
        }

        const u = u_opt orelse break;
        visited[u] = true;

        for (0..n) |v| {
            const weight = graph[u][v];
            if (weight <= 0 or visited[v]) continue;
            const sum = @addWithOverflow(dist[u], weight);
            if (sum[1] != 0) continue;
            if (sum[0] < dist[v]) dist[v] = sum[0];
        }
    }

    return dist;
}

test "dijkstra alternate: python doctest style isolated graph" {
    const alloc = testing.allocator;

    const row0 = [_]i64{ 0, 0, 0, 0 };
    const row1 = [_]i64{ 0, 0, 0, 0 };
    const row2 = [_]i64{ 0, 0, 0, 0 };
    const row3 = [_]i64{ 0, 0, 0, 0 };
    const graph = [_][]const i64{ &row0, &row1, &row2, &row3 };

    const dist = try dijkstraAlternate(alloc, &graph, 1);
    defer alloc.free(dist);

    try testing.expectEqualSlices(i64, &[_]i64{ 10_000_000, 0, 10_000_000, 10_000_000 }, dist);
}

test "dijkstra alternate: weighted example" {
    const alloc = testing.allocator;

    const row0 = [_]i64{ 0, 4, 1, 0, 0 };
    const row1 = [_]i64{ 0, 0, 0, 1, 0 };
    const row2 = [_]i64{ 0, 2, 0, 5, 0 };
    const row3 = [_]i64{ 0, 0, 0, 0, 3 };
    const row4 = [_]i64{ 0, 0, 0, 0, 0 };
    const graph = [_][]const i64{ &row0, &row1, &row2, &row3, &row4 };

    const dist = try dijkstraAlternate(alloc, &graph, 0);
    defer alloc.free(dist);

    try testing.expectEqualSlices(i64, &[_]i64{ 0, 3, 1, 4, 7 }, dist);
}

test "dijkstra alternate: invalid source returns empty" {
    const alloc = testing.allocator;
    const row0 = [_]i64{0};
    const graph = [_][]const i64{&row0};

    const dist = try dijkstraAlternate(alloc, &graph, 3);
    defer alloc.free(dist);
    try testing.expectEqual(@as(usize, 0), dist.len);
}

test "dijkstra alternate: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 180;

    const storage = try alloc.alloc(i64, n * n);
    defer alloc.free(storage);
    @memset(storage, 0);

    const rows = try alloc.alloc([]const i64, n);
    defer alloc.free(rows);

    for (0..n - 1) |i| storage[i * n + i + 1] = 1;
    for (0..n) |i| rows[i] = storage[i * n .. (i + 1) * n];

    const dist = try dijkstraAlternate(alloc, rows, 0);
    defer alloc.free(dist);

    try testing.expectEqual(@as(i64, @intCast(n - 1)), dist[n - 1]);
}
