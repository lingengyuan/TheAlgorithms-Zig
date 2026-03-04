//! Articulation Points (Undirected Graph) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/articulation_points.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns articulation points in ascending order for an undirected graph.
/// Graph is adjacency-list based; invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn articulationPoints(allocator: Allocator, adj: []const []const usize) ![]usize {
    const n = adj.len;
    if (n == 0) return try allocator.alloc(usize, 0);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    const disc = try allocator.alloc(usize, n);
    defer allocator.free(disc);
    const low = try allocator.alloc(usize, n);
    defer allocator.free(low);
    const parent = try allocator.alloc(usize, n);
    defer allocator.free(parent);
    const is_art = try allocator.alloc(bool, n);
    defer allocator.free(is_art);

    const none = std.math.maxInt(usize);
    @memset(visited, false);
    @memset(disc, 0);
    @memset(low, 0);
    @memset(parent, none);
    @memset(is_art, false);

    var ctx = Context{
        .allocator = allocator,
        .adj = adj,
        .visited = visited,
        .disc = disc,
        .low = low,
        .parent = parent,
        .is_art = is_art,
        .time = 0,
    };

    for (0..n) |v| {
        if (!ctx.visited[v]) {
            try ctx.dfs(v, v);
        }
    }

    var out = std.ArrayListUnmanaged(usize){};
    defer out.deinit(allocator);
    for (0..n) |v| {
        if (is_art[v]) try out.append(allocator, v);
    }
    return try out.toOwnedSlice(allocator);
}

const Context = struct {
    allocator: Allocator,
    adj: []const []const usize,
    visited: []bool,
    disc: []usize,
    low: []usize,
    parent: []usize,
    is_art: []bool,
    time: usize,

    fn dfs(self: *Context, root: usize, at: usize) !void {
        self.visited[at] = true;
        self.disc[at] = self.time;
        self.low[at] = self.time;
        self.time += 1;

        var root_children: usize = 0;

        for (self.adj[at]) |to| {
            if (to >= self.adj.len) continue;

            if (!self.visited[to]) {
                self.parent[to] = at;
                if (at == root) root_children += 1;

                try self.dfs(root, to);

                if (self.low[to] < self.low[at]) {
                    self.low[at] = self.low[to];
                }

                if (at != root and self.low[to] >= self.disc[at]) {
                    self.is_art[at] = true;
                }
            } else if (to != self.parent[at]) {
                if (self.disc[to] < self.low[at]) {
                    self.low[at] = self.disc[to];
                }
            }
        }

        if (at == root and root_children > 1) {
            self.is_art[at] = true;
        }
    }
};

test "articulation points: python sample graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{ 0, 2 }, // 1
        &[_]usize{ 0, 1, 3, 5 }, // 2
        &[_]usize{ 2, 4 }, // 3
        &[_]usize{3}, // 4
        &[_]usize{ 2, 6, 8 }, // 5
        &[_]usize{ 5, 7 }, // 6
        &[_]usize{ 6, 8 }, // 7
        &[_]usize{ 5, 7 }, // 8
    };

    const result = try articulationPoints(alloc, &adj);
    defer alloc.free(result);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 5 }, result);
}

test "articulation points: cycle graph has none" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 1, 3 },
        &[_]usize{ 2, 0 },
    };

    const result = try articulationPoints(alloc, &adj);
    defer alloc.free(result);
    try testing.expectEqual(@as(usize, 0), result.len);
}

test "articulation points: disconnected graph and invalid neighbors" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1}, // component A
        &[_]usize{ 0, 2, 99 },
        &[_]usize{1},
        &[_]usize{4}, // component B
        &[_]usize{3},
    };

    const result = try articulationPoints(alloc, &adj);
    defer alloc.free(result);
    try testing.expectEqualSlices(usize, &[_]usize{1}, result);
}

test "articulation points: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 128;

    const mutable_adj = try alloc.alloc([]usize, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }
    for (0..n) |i| {
        if (i == 0) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = 1;
        } else if (i + 1 == n) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = i - 1;
        } else {
            mutable_adj[i] = try alloc.alloc(usize, 2);
            mutable_adj[i][0] = i - 1;
            mutable_adj[i][1] = i + 1;
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    const result = try articulationPoints(alloc, adj);
    defer alloc.free(result);

    try testing.expectEqual(n - 2, result.len);
    for (result, 0..) |v, i| {
        try testing.expectEqual(i + 1, v);
    }
}
