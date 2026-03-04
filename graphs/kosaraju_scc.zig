//! Kosaraju Strongly Connected Components - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/scc_kosaraju.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const KosarajuResult = struct {
    components: [][]usize,

    pub fn deinit(self: KosarajuResult, allocator: Allocator) void {
        for (self.components) |component| allocator.free(component);
        allocator.free(self.components);
    }
};

/// Finds all strongly connected components in a directed graph using Kosaraju's algorithm.
/// Graph is adjacency-list based; invalid neighbor indices are ignored.
/// Returns components in reverse topological order of SCC condensation.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn kosarajuScc(allocator: Allocator, adj: []const []const usize) !KosarajuResult {
    const n = adj.len;

    const rev = try allocator.alloc(std.ArrayListUnmanaged(usize), n);
    defer {
        for (0..n) |i| rev[i].deinit(allocator);
        allocator.free(rev);
    }
    for (0..n) |i| rev[i] = .{};

    for (adj, 0..) |neighbors, u| {
        for (neighbors) |v| {
            if (v >= n) continue;
            try rev[v].append(allocator, u);
        }
    }

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var order = std.ArrayListUnmanaged(usize){};
    defer order.deinit(allocator);

    var first_pass = FirstPass{
        .allocator = allocator,
        .adj = adj,
        .visited = visited,
        .order = &order,
    };

    for (0..n) |v| {
        if (!visited[v]) {
            try first_pass.dfs(v);
        }
    }

    @memset(visited, false);

    var components = std.ArrayListUnmanaged([]usize){};
    errdefer {
        for (components.items) |component| allocator.free(component);
        components.deinit(allocator);
    }

    var second_pass = SecondPass{
        .allocator = allocator,
        .rev = rev,
        .visited = visited,
    };

    var i = order.items.len;
    while (i > 0) {
        i -= 1;
        const v = order.items[i];
        if (visited[v]) continue;

        var component = std.ArrayListUnmanaged(usize){};
        errdefer component.deinit(allocator);
        try second_pass.dfs(v, &component);
        try components.append(allocator, try component.toOwnedSlice(allocator));
    }

    return .{
        .components = try components.toOwnedSlice(allocator),
    };
}

const FirstPass = struct {
    allocator: Allocator,
    adj: []const []const usize,
    visited: []bool,
    order: *std.ArrayListUnmanaged(usize),

    fn dfs(self: *FirstPass, u: usize) !void {
        self.visited[u] = true;
        for (self.adj[u]) |v| {
            if (v >= self.adj.len) continue;
            if (!self.visited[v]) {
                try self.dfs(v);
            }
        }
        try self.order.append(self.allocator, u);
    }
};

const SecondPass = struct {
    allocator: Allocator,
    rev: []const std.ArrayListUnmanaged(usize),
    visited: []bool,

    fn dfs(self: *SecondPass, u: usize, component: *std.ArrayListUnmanaged(usize)) !void {
        self.visited[u] = true;
        try component.append(self.allocator, u);

        for (self.rev[u].items) |v| {
            if (!self.visited[v]) {
                try self.dfs(v, component);
            }
        }
    }
};

fn sortComponent(component: []usize) void {
    std.sort.heap(usize, component, {}, comptime std.sort.asc(usize));
}

fn lessComponents(_: void, a: []usize, b: []usize) bool {
    if (a.len == 0 or b.len == 0) return a.len < b.len;
    if (a[0] != b[0]) return a[0] < b[0];
    if (a.len != b.len) return a.len < b.len;
    for (0..a.len) |i| {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return false;
}

fn normalize(components: [][]usize) void {
    for (components) |component| sortComponent(component);
    std.sort.heap([]usize, components, {}, lessComponents);
}

fn expectComponents(expected: []const []const usize, actual: [][]usize) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, 0..) |component, i| {
        try testing.expectEqualSlices(usize, component, actual[i]);
    }
}

test "kosaraju scc: sample directed graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{ 0, 3 },
        &[_]usize{4},
        &[_]usize{ 5, 7 },
        &[_]usize{6},
        &[_]usize{4},
        &[_]usize{},
    };

    var result = try kosarajuScc(alloc, &adj);
    defer result.deinit(alloc);
    normalize(result.components);

    const expected = [_][]const usize{
        &[_]usize{ 0, 1, 2 },
        &[_]usize{3},
        &[_]usize{ 4, 5, 6 },
        &[_]usize{7},
    };
    try expectComponents(&expected, result.components);
}

test "kosaraju scc: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};

    var result = try kosarajuScc(alloc, &adj);
    defer result.deinit(alloc);
    try testing.expectEqual(@as(usize, 0), result.components.len);
}

test "kosaraju scc: invalid neighbor indices are ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 8 },
        &[_]usize{0},
        &[_]usize{},
    };

    var result = try kosarajuScc(alloc, &adj);
    defer result.deinit(alloc);
    normalize(result.components);

    const expected = [_][]const usize{
        &[_]usize{ 0, 1 },
        &[_]usize{2},
    };
    try expectComponents(&expected, result.components);
}

test "kosaraju scc: extreme chain produces singleton SCCs" {
    const alloc = testing.allocator;
    const n: usize = 128;

    const mutable_adj = try alloc.alloc([]usize, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }

    for (0..n) |i| {
        if (i + 1 < n) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = i + 1;
        } else {
            mutable_adj[i] = try alloc.alloc(usize, 0);
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    var result = try kosarajuScc(alloc, adj);
    defer result.deinit(alloc);
    normalize(result.components);

    try testing.expectEqual(n, result.components.len);
    for (result.components, 0..) |component, i| {
        try testing.expectEqual(@as(usize, 1), component.len);
        try testing.expectEqual(i, component[0]);
    }
}
