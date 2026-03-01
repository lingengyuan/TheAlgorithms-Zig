//! Bridges in Undirected Graph - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/finding_bridges.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Bridge = struct {
    u: usize,
    v: usize,
};

/// Finds all bridges in an undirected graph.
/// Graph is adjacency-list based; invalid neighbor indices are ignored.
/// Returned bridge endpoints satisfy `u < v`.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn findBridges(allocator: Allocator, adj: []const []const usize) ![]Bridge {
    const n = adj.len;
    if (n == 0) return try allocator.alloc(Bridge, 0);

    const none = std.math.maxInt(usize);
    const disc = try allocator.alloc(usize, n);
    defer allocator.free(disc);
    const low = try allocator.alloc(usize, n);
    defer allocator.free(low);
    @memset(disc, none);
    @memset(low, 0);

    var ctx = Context{
        .allocator = allocator,
        .adj = adj,
        .disc = disc,
        .low = low,
        .time = 0,
        .bridges = .{},
    };
    errdefer ctx.bridges.deinit(allocator);

    for (0..n) |node| {
        if (ctx.disc[node] == none) {
            try ctx.dfs(node, none);
        }
    }

    const out = try ctx.bridges.toOwnedSlice(allocator);
    sortBridges(out);
    return out;
}

const Context = struct {
    allocator: Allocator,
    adj: []const []const usize,
    disc: []usize,
    low: []usize,
    time: usize,
    bridges: std.ArrayListUnmanaged(Bridge),

    fn dfs(self: *Context, node: usize, parent: usize) !void {
        self.disc[node] = self.time;
        self.low[node] = self.time;
        self.time += 1;

        var skipped_parent_once = false;

        for (self.adj[node]) |nb| {
            if (nb >= self.adj.len) continue;

            if (nb == parent and !skipped_parent_once) {
                skipped_parent_once = true;
                continue;
            }

            if (self.disc[nb] == std.math.maxInt(usize)) {
                try self.dfs(nb, node);
                if (self.low[nb] < self.low[node]) {
                    self.low[node] = self.low[nb];
                }
                if (self.low[nb] > self.disc[node]) {
                    const a = @min(node, nb);
                    const b = @max(node, nb);
                    try self.bridges.append(self.allocator, .{ .u = a, .v = b });
                }
            } else if (self.disc[nb] < self.low[node]) {
                self.low[node] = self.disc[nb];
            }
        }
    }
};

fn lessBridge(_: void, a: Bridge, b: Bridge) bool {
    if (a.u != b.u) return a.u < b.u;
    return a.v < b.v;
}

fn sortBridges(bridges: []Bridge) void {
    std.sort.heap(Bridge, bridges, {}, lessBridge);
}

test "bridges: sample graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1, 3, 5 },
        &[_]usize{ 2, 4 },
        &[_]usize{3},
        &[_]usize{ 2, 6, 8 },
        &[_]usize{ 5, 7 },
        &[_]usize{ 6, 8 },
        &[_]usize{ 5, 7 },
    };

    const bridges = try findBridges(alloc, &adj);
    defer alloc.free(bridges);

    try testing.expectEqual(@as(usize, 3), bridges.len);
    try testing.expectEqualDeep(Bridge{ .u = 2, .v = 3 }, bridges[0]);
    try testing.expectEqualDeep(Bridge{ .u = 2, .v = 5 }, bridges[1]);
    try testing.expectEqualDeep(Bridge{ .u = 3, .v = 4 }, bridges[2]);
}

test "bridges: no bridges in cycle" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 2 },
    };

    const bridges = try findBridges(alloc, &adj);
    defer alloc.free(bridges);
    try testing.expectEqual(@as(usize, 0), bridges.len);
}

test "bridges: disconnected graph with isolated node" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{3},
        &[_]usize{2},
        &[_]usize{},
    };

    const bridges = try findBridges(alloc, &adj);
    defer alloc.free(bridges);

    try testing.expectEqual(@as(usize, 2), bridges.len);
    try testing.expectEqualDeep(Bridge{ .u = 0, .v = 1 }, bridges[0]);
    try testing.expectEqualDeep(Bridge{ .u = 2, .v = 3 }, bridges[1]);
}

test "bridges: invalid neighbor indices are ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{0},
    };

    const bridges = try findBridges(alloc, &adj);
    defer alloc.free(bridges);

    try testing.expectEqual(@as(usize, 1), bridges.len);
    try testing.expectEqualDeep(Bridge{ .u = 0, .v = 1 }, bridges[0]);
}

test "bridges: parallel edges are not bridges" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 1 },
        &[_]usize{ 0, 0 },
    };

    const bridges = try findBridges(alloc, &adj);
    defer alloc.free(bridges);
    try testing.expectEqual(@as(usize, 0), bridges.len);
}

test "bridges: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};

    const bridges = try findBridges(alloc, &adj);
    defer alloc.free(bridges);
    try testing.expectEqual(@as(usize, 0), bridges.len);
}
