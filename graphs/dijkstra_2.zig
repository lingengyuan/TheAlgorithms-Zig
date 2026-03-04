//! Dijkstra (Adjacency Matrix Float Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra_2.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes shortest-path distances from `start` on a matrix graph.
/// `inf` means no direct edge. Matrix must be square.
/// Returns empty slice when `start` is out of range.
/// Time complexity: O(V^2), Space complexity: O(V)
pub fn dijkstraMatrixFloat(allocator: Allocator, matrix: []const []const f64, start: usize) ![]f64 {
    const n = matrix.len;
    if (start >= n) return try allocator.alloc(f64, 0);

    for (matrix) |row| {
        if (row.len != n) return error.InvalidGraph;
    }

    const inf = std.math.inf(f64);
    const dist = try allocator.alloc(f64, n);
    errdefer allocator.free(dist);
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);

    @memset(dist, inf);
    @memset(visited, false);
    dist[start] = 0.0;

    var iter: usize = 0;
    while (iter < n) : (iter += 1) {
        var best = inf;
        var u_opt: ?usize = null;

        for (0..n) |i| {
            if (!visited[i] and dist[i] < best) {
                best = dist[i];
                u_opt = i;
            }
        }

        const u = u_opt orelse break;
        visited[u] = true;

        for (0..n) |v| {
            if (visited[v]) continue;
            const w = matrix[u][v];
            if (!std.math.isFinite(w)) continue;
            const candidate = dist[u] + w;
            if (candidate < dist[v]) dist[v] = candidate;
        }
    }

    return dist;
}

test "dijkstra 2: basic matrix graph" {
    const alloc = testing.allocator;
    const inf = std.math.inf(f64);

    const row0 = [_]f64{ 0, 4, 1, inf };
    const row1 = [_]f64{ inf, 0, inf, 1 };
    const row2 = [_]f64{ inf, 2, 0, 5 };
    const row3 = [_]f64{ inf, inf, inf, 0 };
    const matrix = [_][]const f64{ &row0, &row1, &row2, &row3 };

    const dist = try dijkstraMatrixFloat(alloc, &matrix, 0);
    defer alloc.free(dist);

    try testing.expectApproxEqAbs(@as(f64, 0), dist[0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3), dist[1], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1), dist[2], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4), dist[3], 1e-9);
}

test "dijkstra 2: invalid start returns empty" {
    const alloc = testing.allocator;
    const row0 = [_]f64{0};
    const matrix = [_][]const f64{&row0};

    const dist = try dijkstraMatrixFloat(alloc, &matrix, 2);
    defer alloc.free(dist);
    try testing.expectEqual(@as(usize, 0), dist.len);
}

test "dijkstra 2: disconnected vertices remain inf" {
    const alloc = testing.allocator;
    const inf = std.math.inf(f64);

    const row0 = [_]f64{ 0, 1, inf };
    const row1 = [_]f64{ inf, 0, inf };
    const row2 = [_]f64{ inf, inf, 0 };
    const matrix = [_][]const f64{ &row0, &row1, &row2 };

    const dist = try dijkstraMatrixFloat(alloc, &matrix, 0);
    defer alloc.free(dist);
    try testing.expect(std.math.isFinite(dist[0]));
    try testing.expect(std.math.isFinite(dist[1]));
    try testing.expect(!std.math.isFinite(dist[2]));
}

test "dijkstra 2: extreme chain graph" {
    const alloc = testing.allocator;
    const n: usize = 129;
    const inf = std.math.inf(f64);

    const matrix_storage = try alloc.alloc(f64, n * n);
    defer alloc.free(matrix_storage);
    @memset(matrix_storage, inf);

    const rows = try alloc.alloc([]const f64, n);
    defer alloc.free(rows);

    for (0..n) |i| {
        matrix_storage[i * n + i] = 0;
        if (i + 1 < n) matrix_storage[i * n + i + 1] = 1;
    }
    for (0..n) |i| rows[i] = matrix_storage[i * n .. (i + 1) * n];

    const dist = try dijkstraMatrixFloat(alloc, rows, 0);
    defer alloc.free(dist);

    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(n - 1)), dist[n - 1], 1e-9);
}
