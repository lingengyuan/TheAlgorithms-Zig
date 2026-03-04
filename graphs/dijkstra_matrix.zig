//! Dijkstra (Adjacency Matrix Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra_2.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes shortest distances from `source` on a weighted directed graph
/// represented by adjacency matrix where `null` means no edge.
/// Returns `std.math.maxInt(i64)` for unreachable vertices.
/// Negative edges are rejected.
/// Time complexity: O(V²), Space complexity: O(V)
pub fn dijkstraMatrix(
    allocator: Allocator,
    matrix: []const []const ?i64,
    source: usize,
) ![]i64 {
    const n = matrix.len;
    if (source >= n) return error.InvalidNode;
    if (n == 0) return error.InvalidNode;

    for (matrix, 0..) |row, i| {
        if (row.len != n) return error.InvalidMatrix;
        if (row[i] != null and row[i].? != 0) return error.InvalidDiagonal;
        for (row) |cell| {
            if (cell) |w| {
                if (w < 0) return error.NegativeWeight;
            }
        }
    }

    const inf: i64 = std.math.maxInt(i64);
    const dist = try allocator.alloc(i64, n);
    errdefer allocator.free(dist);
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(dist, inf);
    @memset(visited, false);
    dist[source] = 0;

    for (0..n) |_| {
        var min_idx: ?usize = null;
        var min_val: i64 = inf;
        for (0..n) |i| {
            if (!visited[i] and dist[i] < min_val) {
                min_val = dist[i];
                min_idx = i;
            }
        }
        const u = min_idx orelse break;
        visited[u] = true;

        for (0..n) |v| {
            if (visited[v]) continue;
            const w_opt = matrix[u][v];
            if (w_opt == null) continue;
            const w = w_opt.?;
            if (dist[u] == inf) continue;
            const sum = @addWithOverflow(dist[u], w);
            if (sum[1] != 0) return error.Overflow;
            if (sum[0] < dist[v]) dist[v] = sum[0];
        }
    }

    return dist;
}

test "dijkstra matrix: basic directed graph" {
    const alloc = testing.allocator;
    const m = [_][]const ?i64{
        &[_]?i64{ 0, 4, 1, null },
        &[_]?i64{ null, 0, null, 1 },
        &[_]?i64{ null, 2, 0, 5 },
        &[_]?i64{ null, null, null, 0 },
    };

    const dist = try dijkstraMatrix(alloc, &m, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 3, 1, 4 }, dist);
}

test "dijkstra matrix: disconnected and invalid input" {
    const alloc = testing.allocator;
    const inf = std.math.maxInt(i64);

    const m = [_][]const ?i64{
        &[_]?i64{ 0, 3, null },
        &[_]?i64{ null, 0, null },
        &[_]?i64{ null, null, 0 },
    };
    const dist = try dijkstraMatrix(alloc, &m, 0);
    defer alloc.free(dist);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 3, inf }, dist);

    try testing.expectError(error.InvalidNode, dijkstraMatrix(alloc, &m, 7));

    const ragged = [_][]const ?i64{
        &[_]?i64{ 0, 1 },
        &[_]?i64{0},
    };
    try testing.expectError(error.InvalidMatrix, dijkstraMatrix(alloc, &ragged, 0));
}

test "dijkstra matrix: negative weight and invalid diagonal" {
    const alloc = testing.allocator;
    const neg = [_][]const ?i64{
        &[_]?i64{ 0, -1 },
        &[_]?i64{ null, 0 },
    };
    try testing.expectError(error.NegativeWeight, dijkstraMatrix(alloc, &neg, 0));

    const bad_diag = [_][]const ?i64{
        &[_]?i64{ 1, 2 },
        &[_]?i64{ 3, 0 },
    };
    try testing.expectError(error.InvalidDiagonal, dijkstraMatrix(alloc, &bad_diag, 0));
}

test "dijkstra matrix: extreme dense graph" {
    const alloc = testing.allocator;
    const n: usize = 90;
    const data = try alloc.alloc(?i64, n * n);
    defer alloc.free(data);

    for (0..n) |i| {
        for (0..n) |j| {
            data[i * n + j] = if (i == j) 0 else 1;
        }
    }

    const rows = try alloc.alloc([]const ?i64, n);
    defer alloc.free(rows);
    for (0..n) |i| rows[i] = data[i * n .. (i + 1) * n];

    const dist = try dijkstraMatrix(alloc, rows, 0);
    defer alloc.free(dist);

    try testing.expectEqual(@as(i64, 0), dist[0]);
    for (1..n) |i| try testing.expectEqual(@as(i64, 1), dist[i]);
}
