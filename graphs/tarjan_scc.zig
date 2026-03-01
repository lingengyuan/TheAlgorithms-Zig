//! Tarjan Strongly Connected Components - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/tarjans_scc.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const TarjanResult = struct {
    components: [][]usize,

    pub fn deinit(self: TarjanResult, allocator: Allocator) void {
        for (self.components) |component| allocator.free(component);
        allocator.free(self.components);
    }
};

/// Finds all strongly connected components in a directed graph using Tarjan's algorithm.
/// Graph is adjacency-list based; invalid neighbor indices are ignored.
/// Returns components as a list of node-id slices.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn tarjanScc(allocator: Allocator, adj: []const []const usize) !TarjanResult {
    const n = adj.len;
    const none = std.math.maxInt(usize);

    const indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);
    const lowlink = try allocator.alloc(usize, n);
    defer allocator.free(lowlink);
    const on_stack = try allocator.alloc(bool, n);
    defer allocator.free(on_stack);

    @memset(indices, none);
    @memset(lowlink, 0);
    @memset(on_stack, false);

    var ctx = Context{
        .allocator = allocator,
        .adj = adj,
        .indices = indices,
        .lowlink = lowlink,
        .on_stack = on_stack,
        .stack = .{},
        .next_index = 0,
        .components = .{},
    };
    defer ctx.stack.deinit(allocator);
    errdefer {
        for (ctx.components.items) |component| allocator.free(component);
        ctx.components.deinit(allocator);
    }

    for (0..n) |v| {
        if (ctx.indices[v] == none) {
            try ctx.strongConnect(v);
        }
    }

    return .{
        .components = try ctx.components.toOwnedSlice(allocator),
    };
}

const Context = struct {
    allocator: Allocator,
    adj: []const []const usize,
    indices: []usize,
    lowlink: []usize,
    on_stack: []bool,
    stack: std.ArrayListUnmanaged(usize),
    next_index: usize,
    components: std.ArrayListUnmanaged([]usize),

    fn strongConnect(self: *Context, v: usize) !void {
        const none = std.math.maxInt(usize);

        self.indices[v] = self.next_index;
        self.lowlink[v] = self.next_index;
        self.next_index += 1;
        try self.stack.append(self.allocator, v);
        self.on_stack[v] = true;

        for (self.adj[v]) |w| {
            if (w >= self.adj.len) continue;

            if (self.indices[w] == none) {
                try self.strongConnect(w);
                if (self.lowlink[w] < self.lowlink[v]) {
                    self.lowlink[v] = self.lowlink[w];
                }
            } else if (self.on_stack[w] and self.indices[w] < self.lowlink[v]) {
                self.lowlink[v] = self.indices[w];
            }
        }

        if (self.lowlink[v] == self.indices[v]) {
            var component = std.ArrayListUnmanaged(usize){};
            errdefer component.deinit(self.allocator);

            while (true) {
                const node = self.stack.pop() orelse return error.InternalInvariantBroken;
                self.on_stack[node] = false;
                try component.append(self.allocator, node);
                if (node == v) break;
            }

            try self.components.append(self.allocator, try component.toOwnedSlice(self.allocator));
        }
    }
};

fn sortComponentNodes(component: []usize) void {
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

fn normalizeComponents(components: [][]usize) void {
    for (components) |component| {
        sortComponentNodes(component);
    }
    std.sort.heap([]usize, components, {}, lessComponents);
}

fn expectComponentsEqual(expected: []const []const usize, actual: [][]usize) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, 0..) |expected_component, i| {
        try testing.expectEqualSlices(usize, expected_component, actual[i]);
    }
}

test "tarjan scc: basic directed graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1}, // 0
        &[_]usize{ 2, 3 }, // 1
        &[_]usize{0}, // 2
        &[_]usize{4}, // 3
        &[_]usize{ 3, 5 }, // 4
        &[_]usize{}, // 5
    };

    var result = try tarjanScc(alloc, &adj);
    defer result.deinit(alloc);
    normalizeComponents(result.components);

    const expected = [_][]const usize{
        &[_]usize{ 0, 1, 2 },
        &[_]usize{ 3, 4 },
        &[_]usize{5},
    };
    try expectComponentsEqual(&expected, result.components);
}

test "tarjan scc: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};

    var result = try tarjanScc(alloc, &adj);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 0), result.components.len);
}

test "tarjan scc: single node graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{},
    };

    var result = try tarjanScc(alloc, &adj);
    defer result.deinit(alloc);
    normalizeComponents(result.components);

    const expected = [_][]const usize{
        &[_]usize{0},
    };
    try expectComponentsEqual(&expected, result.components);
}

test "tarjan scc: invalid neighbor index is ignored and self-loop works" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 0, 9 },
        &[_]usize{2},
        &[_]usize{1},
    };

    var result = try tarjanScc(alloc, &adj);
    defer result.deinit(alloc);
    normalizeComponents(result.components);

    const expected = [_][]const usize{
        &[_]usize{0},
        &[_]usize{ 1, 2 },
    };
    try expectComponentsEqual(&expected, result.components);
}

test "tarjan scc: disconnected graph with isolated nodes" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{},
        &[_]usize{4},
        &[_]usize{},
    };

    var result = try tarjanScc(alloc, &adj);
    defer result.deinit(alloc);
    normalizeComponents(result.components);

    const expected = [_][]const usize{
        &[_]usize{ 0, 1 },
        &[_]usize{2},
        &[_]usize{3},
        &[_]usize{4},
    };
    try expectComponentsEqual(&expected, result.components);
}

test "tarjan scc: degenerate long chain produces singleton components" {
    const alloc = testing.allocator;
    const n: usize = 64;

    const adj_mut = try alloc.alloc([]usize, n);
    defer {
        for (adj_mut) |row| alloc.free(row);
        alloc.free(adj_mut);
    }
    for (0..n) |i| {
        if (i + 1 < n) {
            adj_mut[i] = try alloc.alloc(usize, 1);
            adj_mut[i][0] = i + 1;
        } else {
            adj_mut[i] = try alloc.alloc(usize, 0);
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (adj_mut, 0..) |row, i| adj[i] = row;

    var result = try tarjanScc(alloc, adj);
    defer result.deinit(alloc);
    normalizeComponents(result.components);

    try testing.expectEqual(n, result.components.len);
    for (result.components, 0..) |component, i| {
        try testing.expectEqual(@as(usize, 1), component.len);
        try testing.expectEqual(i, component[0]);
    }
}
