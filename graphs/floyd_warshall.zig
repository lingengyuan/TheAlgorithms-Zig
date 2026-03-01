//! Floyd-Warshall All-Pairs Shortest Paths - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/floyd_warshall.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes all-pairs shortest paths from a flattened `n x n` adjacency matrix.
/// `inf` is the sentinel value for unreachable pairs.
/// Returns a newly allocated flattened matrix; caller owns returned memory.
/// Time complexity: O(V^3), Space complexity: O(V^2)
pub fn floydWarshall(allocator: Allocator, matrix: []const i64, n: usize, inf: i64) ![]i64 {
    if (n == 0) return try allocator.alloc(i64, 0);
    const elem_count = @mulWithOverflow(n, n);
    if (elem_count[1] != 0) return error.Overflow;
    if (matrix.len != elem_count[0]) return error.InvalidMatrixSize;

    const dist = try allocator.dupe(i64, matrix);

    for (0..n) |k| {
        for (0..n) |i| {
            const ik = dist[i * n + k];
            if (ik == inf) continue;

            for (0..n) |j| {
                const kj = dist[k * n + j];
                if (kj == inf) continue;

                const sum = @addWithOverflow(ik, kj);
                if (sum[1] != 0) continue;

                const idx = i * n + j;
                if (sum[0] < dist[idx]) {
                    dist[idx] = sum[0];
                }
            }
        }
    }

    return dist;
}

test "floyd-warshall: basic graph" {
    const alloc = testing.allocator;
    const inf: i64 = 1_000_000_000;
    const n: usize = 4;
    const matrix = [_]i64{
        0,   5,   inf, 10,
        inf, 0,   3,   inf,
        inf, inf, 0,   1,
        inf, inf, inf, 0,
    };

    const out = try floydWarshall(alloc, &matrix, n, inf);
    defer alloc.free(out);

    const expected = [_]i64{
        0,   5,   8,   9,
        inf, 0,   3,   4,
        inf, inf, 0,   1,
        inf, inf, inf, 0,
    };
    try testing.expectEqualSlices(i64, &expected, out);
}

test "floyd-warshall: graph with negative edge and no negative cycle" {
    const alloc = testing.allocator;
    const inf: i64 = 1_000_000_000;
    const n: usize = 3;
    const matrix = [_]i64{
        0,   1,   inf,
        inf, 0,   -1,
        2,   inf, 0,
    };

    const out = try floydWarshall(alloc, &matrix, n, inf);
    defer alloc.free(out);

    const expected = [_]i64{
        0, 1, 0,
        1, 0, -1,
        2, 3, 0,
    };
    try testing.expectEqualSlices(i64, &expected, out);
}

test "floyd-warshall: disconnected graph keeps inf" {
    const alloc = testing.allocator;
    const inf: i64 = 1_000_000_000;
    const n: usize = 2;
    const matrix = [_]i64{
        0,   inf,
        inf, 0,
    };
    const out = try floydWarshall(alloc, &matrix, n, inf);
    defer alloc.free(out);
    try testing.expectEqualSlices(i64, &matrix, out);
}

test "floyd-warshall: invalid matrix size returns error" {
    const alloc = testing.allocator;
    const inf: i64 = 1_000_000_000;
    const matrix = [_]i64{ 0, 1, 2 };
    try testing.expectError(error.InvalidMatrixSize, floydWarshall(alloc, &matrix, 2, inf));
}

test "floyd-warshall: empty graph returns empty" {
    const alloc = testing.allocator;
    const out = try floydWarshall(alloc, &[_]i64{}, 0, 1_000_000_000);
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "floyd-warshall: oversize dimension returns overflow" {
    const alloc = testing.allocator;
    const tiny = [_]i64{0};
    try testing.expectError(error.Overflow, floydWarshall(alloc, &tiny, std.math.maxInt(usize), 1_000_000_000));
}
