//! PageRank (Iterative) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/page_rank.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes PageRank scores with in-place iterative updates matching the Python reference.
/// Invalid neighbor indices are ignored.
/// Nodes with zero outbound edges contribute nothing to others (safe handling).
/// Time complexity: O(iterations * (V + E)), Space complexity: O(V + E)
pub fn pageRank(
    allocator: Allocator,
    adj: []const []const usize,
    iterations: usize,
    damping: f64,
) ![]f64 {
    if (!(damping >= 0.0 and damping <= 1.0)) return error.InvalidDamping;

    const n = adj.len;
    if (n == 0) return try allocator.alloc(f64, 0);

    const inbound = try allocator.alloc(std.ArrayListUnmanaged(usize), n);
    defer {
        for (inbound) |*list| list.deinit(allocator);
        allocator.free(inbound);
    }
    for (inbound) |*list| list.* = .{};

    const out_degree = try allocator.alloc(usize, n);
    defer allocator.free(out_degree);
    @memset(out_degree, 0);

    for (adj, 0..) |neighbors, u| {
        for (neighbors) |v| {
            if (v >= n) continue;
            try inbound[v].append(allocator, u);
            const sum = @addWithOverflow(out_degree[u], 1);
            if (sum[1] != 0) return error.Overflow;
            out_degree[u] = sum[0];
        }
    }

    const ranks = try allocator.alloc(f64, n);
    @memset(ranks, 1.0);

    for (0..iterations) |_| {
        for (0..n) |node| {
            var contribution: f64 = 0.0;
            for (inbound[node].items) |from| {
                const out = out_degree[from];
                if (out == 0) continue;
                contribution += ranks[from] / @as(f64, @floatFromInt(out));
            }
            ranks[node] = (1.0 - damping) + damping * contribution;
        }
    }

    return ranks;
}

test "page rank: python sample matrix values after 3 iterations" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // A
        &[_]usize{2}, // B
        &[_]usize{0}, // C
    };

    const ranks = try pageRank(alloc, &adj, 3, 0.85);
    defer alloc.free(ranks);

    try testing.expectEqual(@as(usize, 3), ranks.len);
    try testing.expectApproxEqAbs(@as(f64, 1.09040168359375), ranks[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.6134207155273438), ranks[1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.134828323725586), ranks[2], 1e-12);
}

test "page rank: edge cases and invalid damping" {
    const alloc = testing.allocator;

    const empty = [_][]const usize{};
    const empty_ranks = try pageRank(alloc, &empty, 5, 0.85);
    defer alloc.free(empty_ranks);
    try testing.expectEqual(@as(usize, 0), empty_ranks.len);

    const adj = [_][]const usize{
        &[_]usize{ 1, 99 }, // invalid target is ignored
        &[_]usize{},
    };
    const ranks = try pageRank(alloc, &adj, 2, 0.85);
    defer alloc.free(ranks);
    try testing.expectEqual(@as(usize, 2), ranks.len);

    try testing.expectError(error.InvalidDamping, pageRank(alloc, &adj, 2, -0.1));
    try testing.expectError(error.InvalidDamping, pageRank(alloc, &adj, 2, 1.1));
}

test "page rank: extreme ring graph keeps uniform ranks" {
    const alloc = testing.allocator;
    const n: usize = 128;

    const mutable_adj = try alloc.alloc([]usize, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }
    for (0..n) |i| {
        mutable_adj[i] = try alloc.alloc(usize, 1);
        mutable_adj[i][0] = (i + 1) % n;
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    const ranks = try pageRank(alloc, adj, 50, 0.85);
    defer alloc.free(ranks);
    try testing.expectEqual(n, ranks.len);
    for (ranks) |rank| {
        try testing.expectApproxEqAbs(@as(f64, 1.0), rank, 1e-12);
    }
}
