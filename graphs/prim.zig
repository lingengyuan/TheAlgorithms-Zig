//! Prim Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/prim.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Edge = struct {
    to: usize,
    weight: i64,
};

/// Computes MST total weight from `start` using Prim's algorithm.
/// Graph should be undirected and connected for a full MST.
/// Invalid neighbor indices are ignored.
/// Returns `error.DisconnectedGraph` if all vertices cannot be spanned.
/// Time complexity: O(V^2 + E), Space complexity: O(V)
pub fn primMstWeight(allocator: Allocator, adj: []const []const Edge, start: usize) !i64 {
    const n = adj.len;
    if (n == 0) return 0;
    if (start >= n) return error.InvalidStart;

    const inf: i64 = std.math.maxInt(i64);
    const key = try allocator.alloc(i64, n);
    defer allocator.free(key);
    const used = try allocator.alloc(bool, n);
    defer allocator.free(used);

    @memset(key, inf);
    @memset(used, false);
    key[start] = 0;

    var total: i64 = 0;

    for (0..n) |_| {
        var best: i64 = inf;
        var best_idx: ?usize = null;

        for (0..n) |i| {
            if (!used[i] and key[i] < best) {
                best = key[i];
                best_idx = i;
            }
        }

        const u = best_idx orelse return error.DisconnectedGraph;
        if (key[u] == inf) return error.DisconnectedGraph;

        used[u] = true;
        const sum = @addWithOverflow(total, key[u]);
        if (sum[1] != 0) return error.Overflow;
        total = sum[0];

        for (adj[u]) |edge| {
            if (edge.to >= n) continue;
            if (!used[edge.to] and edge.weight < key[edge.to]) {
                key[edge.to] = edge.weight;
            }
        }
    }

    return total;
}

test "prim: basic mst weight" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 10 }, .{ .to = 2, .weight = 6 }, .{ .to = 3, .weight = 5 } },
        &[_]Edge{ .{ .to = 0, .weight = 10 }, .{ .to = 3, .weight = 15 } },
        &[_]Edge{ .{ .to = 0, .weight = 6 }, .{ .to = 3, .weight = 4 } },
        &[_]Edge{ .{ .to = 0, .weight = 5 }, .{ .to = 1, .weight = 15 }, .{ .to = 2, .weight = 4 } },
    };
    try testing.expectEqual(@as(i64, 19), try primMstWeight(alloc, &adj, 0));
}

test "prim: disconnected graph returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{.{ .to = 1, .weight = 1 }},
        &[_]Edge{.{ .to = 0, .weight = 1 }},
        &[_]Edge{},
    };
    try testing.expectError(error.DisconnectedGraph, primMstWeight(alloc, &adj, 0));
}

test "prim: invalid start returns error" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{},
    };
    try testing.expectError(error.InvalidStart, primMstWeight(alloc, &adj, 2));
}

test "prim: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 2 }, .{ .to = 9, .weight = 1 } },
        &[_]Edge{.{ .to = 0, .weight = 2 }},
    };
    try testing.expectEqual(@as(i64, 2), try primMstWeight(alloc, &adj, 0));
}

test "prim: empty graph returns zero" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{};
    try testing.expectEqual(@as(i64, 0), try primMstWeight(alloc, &adj, 0));
}
