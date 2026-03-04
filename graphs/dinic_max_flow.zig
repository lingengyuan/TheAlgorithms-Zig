//! Dinic Maximum Flow - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dinic.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CapacityEdge = struct {
    from: usize,
    to: usize,
    capacity: i64,
};

const ResidualEdge = struct {
    to: usize,
    rev: usize,
    cap: i64,
};

/// Computes maximum flow from `source` to `sink` using Dinic's algorithm.
/// Input graph is directed; capacities must be non-negative.
/// Time complexity: O(V^2 * E), Space complexity: O(V + E)
pub fn dinicMaxFlow(
    allocator: Allocator,
    node_count: usize,
    edges: []const CapacityEdge,
    source: usize,
    sink: usize,
) !i64 {
    if (source >= node_count or sink >= node_count) return error.InvalidNode;
    if (source == sink) return 0;

    const graph = try allocator.alloc(std.ArrayListUnmanaged(ResidualEdge), node_count);
    defer {
        for (graph) |*list| list.deinit(allocator);
        allocator.free(graph);
    }
    for (graph) |*list| list.* = .{};

    for (edges) |edge| {
        if (edge.from >= node_count or edge.to >= node_count) return error.InvalidNode;
        if (edge.capacity < 0) return error.NegativeCapacity;
        try addEdge(allocator, graph, edge.from, edge.to, edge.capacity);
    }

    const level = try allocator.alloc(i32, node_count);
    defer allocator.free(level);
    const ptr = try allocator.alloc(usize, node_count);
    defer allocator.free(ptr);

    var flow: i64 = 0;
    while (try buildLevelGraph(allocator, graph, source, sink, level)) {
        @memset(ptr, 0);
        while (true) {
            const pushed = try sendFlow(graph, level, ptr, source, sink, std.math.maxInt(i64));
            if (pushed == 0) break;
            const sum = @addWithOverflow(flow, pushed);
            if (sum[1] != 0) return error.Overflow;
            flow = sum[0];
        }
    }
    return flow;
}

fn addEdge(
    allocator: Allocator,
    graph: []std.ArrayListUnmanaged(ResidualEdge),
    from: usize,
    to: usize,
    cap: i64,
) !void {
    const from_rev = graph[from].items.len;
    const to_rev = graph[to].items.len;

    try graph[from].append(allocator, .{
        .to = to,
        .rev = to_rev,
        .cap = cap,
    });
    try graph[to].append(allocator, .{
        .to = from,
        .rev = from_rev,
        .cap = 0,
    });
}

fn buildLevelGraph(
    allocator: Allocator,
    graph: []const std.ArrayListUnmanaged(ResidualEdge),
    source: usize,
    sink: usize,
    level: []i32,
) !bool {
    @memset(level, -1);
    level[source] = 0;

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    var head: usize = 0;
    try queue.append(allocator, source);

    while (head < queue.items.len) {
        const v = queue.items[head];
        head += 1;

        for (graph[v].items) |edge| {
            if (edge.cap <= 0) continue;
            if (level[edge.to] != -1) continue;
            level[edge.to] = level[v] + 1;
            try queue.append(allocator, edge.to);
        }
    }

    return level[sink] != -1;
}

fn sendFlow(
    graph: []std.ArrayListUnmanaged(ResidualEdge),
    level: []const i32,
    ptr: []usize,
    v: usize,
    sink: usize,
    pushed: i64,
) !i64 {
    if (pushed == 0) return 0;
    if (v == sink) return pushed;

    while (ptr[v] < graph[v].items.len) {
        const edge_idx = ptr[v];
        const edge = graph[v].items[edge_idx];

        if (edge.cap > 0 and level[edge.to] == level[v] + 1) {
            const try_push = if (pushed < edge.cap) pushed else edge.cap;
            const flow = try sendFlow(graph, level, ptr, edge.to, sink, try_push);
            if (flow > 0) {
                graph[v].items[edge_idx].cap -= flow;
                const rev_cap = graph[edge.to].items[edge.rev].cap;
                const sum = @addWithOverflow(rev_cap, flow);
                if (sum[1] != 0) return error.Overflow;
                graph[edge.to].items[edge.rev].cap = sum[0];
                return flow;
            }
        }
        ptr[v] += 1;
    }
    return 0;
}

test "dinic max flow: classic sample graph" {
    const alloc = testing.allocator;
    const edges = [_]CapacityEdge{
        .{ .from = 0, .to = 1, .capacity = 16 },
        .{ .from = 0, .to = 2, .capacity = 13 },
        .{ .from = 1, .to = 2, .capacity = 10 },
        .{ .from = 2, .to = 1, .capacity = 4 },
        .{ .from = 1, .to = 3, .capacity = 12 },
        .{ .from = 2, .to = 4, .capacity = 14 },
        .{ .from = 3, .to = 2, .capacity = 9 },
        .{ .from = 4, .to = 3, .capacity = 7 },
        .{ .from = 3, .to = 5, .capacity = 20 },
        .{ .from = 4, .to = 5, .capacity = 4 },
    };

    try testing.expectEqual(@as(i64, 23), try dinicMaxFlow(alloc, 6, &edges, 0, 5));
}

test "dinic max flow: source equals sink and unreachable sink" {
    const alloc = testing.allocator;
    const edges = [_]CapacityEdge{
        .{ .from = 0, .to = 1, .capacity = 3 },
        .{ .from = 1, .to = 2, .capacity = 2 },
    };

    try testing.expectEqual(@as(i64, 0), try dinicMaxFlow(alloc, 3, &edges, 1, 1));
    try testing.expectEqual(@as(i64, 0), try dinicMaxFlow(alloc, 4, &edges, 0, 3));
}

test "dinic max flow: invalid node and negative capacity" {
    const alloc = testing.allocator;
    const invalid = [_]CapacityEdge{
        .{ .from = 0, .to = 2, .capacity = 1 },
    };
    try testing.expectError(error.InvalidNode, dinicMaxFlow(alloc, 2, &invalid, 0, 1));

    const negative = [_]CapacityEdge{
        .{ .from = 0, .to = 1, .capacity = -1 },
    };
    try testing.expectError(error.NegativeCapacity, dinicMaxFlow(alloc, 2, &negative, 0, 1));
}

test "dinic max flow: overflow-prone disjoint huge paths" {
    const alloc = testing.allocator;
    const m = std.math.maxInt(i64);
    const edges = [_]CapacityEdge{
        .{ .from = 0, .to = 1, .capacity = m },
        .{ .from = 1, .to = 3, .capacity = m },
        .{ .from = 0, .to = 2, .capacity = m },
        .{ .from = 2, .to = 3, .capacity = m },
    };

    try testing.expectError(error.Overflow, dinicMaxFlow(alloc, 4, &edges, 0, 3));
}

test "dinic max flow: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 128;
    const edges = try alloc.alloc(CapacityEdge, n - 1);
    defer alloc.free(edges);

    for (0..n - 1) |i| {
        edges[i] = .{ .from = i, .to = i + 1, .capacity = 1 };
    }

    try testing.expectEqual(@as(i64, 1), try dinicMaxFlow(alloc, n, edges, 0, n - 1));
}
