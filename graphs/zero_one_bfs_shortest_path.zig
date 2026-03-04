//! 0-1 BFS Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search_zero_one_shortest_path.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Edge = struct {
    to: usize,
    weight: u8, // must be 0 or 1
};

/// Returns shortest distance from `start` to `finish` in a 0-1 weighted directed graph.
/// Invalid neighbor indices are ignored.
/// Returns `error.NoPath` if unreachable.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn shortestPath01(
    allocator: Allocator,
    adj: []const []const Edge,
    start: usize,
    finish: usize,
) !usize {
    const n = adj.len;
    if (start >= n or finish >= n) return error.InvalidNode;
    if (start == finish) return 0;

    var total_valid_edges: usize = 0;
    for (adj) |neighbors| {
        for (neighbors) |edge| {
            if (edge.weight > 1) return error.InvalidWeight;
            if (edge.to < n) {
                const sum = @addWithOverflow(total_valid_edges, 1);
                if (sum[1] != 0) return error.Overflow;
                total_valid_edges = sum[0];
            }
        }
    }

    const two_e = @mulWithOverflow(total_valid_edges, 2);
    if (two_e[1] != 0) return error.Overflow;
    const cap_base = @addWithOverflow(two_e[0], n);
    if (cap_base[1] != 0) return error.Overflow;
    const cap_plus = @addWithOverflow(cap_base[0], 1);
    if (cap_plus[1] != 0 or cap_plus[0] == 0) return error.Overflow;
    const queue_capacity = cap_plus[0];

    const dist = try allocator.alloc(usize, n);
    defer allocator.free(dist);
    @memset(dist, std.math.maxInt(usize));
    dist[start] = 0;

    var deque = try RingDeque.init(allocator, queue_capacity);
    defer deque.deinit(allocator);
    try deque.pushFront(start);

    while (!deque.isEmpty()) {
        const u = deque.popFront() orelse return error.InternalInvariantBroken;
        const du = dist[u];

        for (adj[u]) |edge| {
            if (edge.weight > 1) return error.InvalidWeight;
            if (edge.to >= n) continue;

            const new_dist_sum = @addWithOverflow(du, @as(usize, @intCast(edge.weight)));
            if (new_dist_sum[1] != 0) return error.Overflow;
            const new_dist = new_dist_sum[0];

            if (new_dist < dist[edge.to]) {
                dist[edge.to] = new_dist;
                if (edge.weight == 0) {
                    try deque.pushFront(edge.to);
                } else {
                    try deque.pushBack(edge.to);
                }
            }
        }
    }

    if (dist[finish] == std.math.maxInt(usize)) return error.NoPath;
    return dist[finish];
}

const RingDeque = struct {
    data: []usize,
    head: usize,
    len: usize,
    allocator: Allocator,

    fn init(allocator: Allocator, capacity: usize) !RingDeque {
        const data = try allocator.alloc(usize, capacity);
        return .{
            .data = data,
            .head = 0,
            .len = 0,
            .allocator = allocator,
        };
    }

    fn deinit(self: *RingDeque, allocator: Allocator) void {
        allocator.free(self.data);
    }

    fn isEmpty(self: *const RingDeque) bool {
        return self.len == 0;
    }

    fn pushFront(self: *RingDeque, value: usize) !void {
        if (self.len == self.data.len) return error.QueueOverflow;
        self.head = if (self.head == 0) self.data.len - 1 else self.head - 1;
        self.data[self.head] = value;
        self.len += 1;
    }

    fn pushBack(self: *RingDeque, value: usize) !void {
        if (self.len == self.data.len) return error.QueueOverflow;
        const tail = (self.head + self.len) % self.data.len;
        self.data[tail] = value;
        self.len += 1;
    }

    fn popFront(self: *RingDeque) ?usize {
        if (self.len == 0) return null;
        const value = self.data[self.head];
        self.head = (self.head + 1) % self.data.len;
        self.len -= 1;
        return value;
    }
};

test "0-1 bfs: python sample graph distances" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{ .{ .to = 1, .weight = 0 }, .{ .to = 3, .weight = 1 } },
        &[_]Edge{.{ .to = 2, .weight = 0 }},
        &[_]Edge{.{ .to = 3, .weight = 0 }},
        &[_]Edge{},
        &[_]Edge{ .{ .to = 2, .weight = 1 }, .{ .to = 5, .weight = 1 }, .{ .to = 6, .weight = 1 } },
        &[_]Edge{.{ .to = 9, .weight = 0 }},
        &[_]Edge{.{ .to = 7, .weight = 1 }},
        &[_]Edge{.{ .to = 8, .weight = 1 }},
        &[_]Edge{.{ .to = 10, .weight = 1 }},
        &[_]Edge{ .{ .to = 7, .weight = 0 }, .{ .to = 10, .weight = 1 } },
        &[_]Edge{},
    };

    try testing.expectEqual(@as(usize, 0), try shortestPath01(alloc, &adj, 0, 3));
    try testing.expectEqual(@as(usize, 2), try shortestPath01(alloc, &adj, 4, 10));
    try testing.expectEqual(@as(usize, 2), try shortestPath01(alloc, &adj, 4, 8));
}

test "0-1 bfs: no path and same node" {
    const alloc = testing.allocator;
    const adj = [_][]const Edge{
        &[_]Edge{.{ .to = 1, .weight = 0 }},
        &[_]Edge{},
        &[_]Edge{},
    };

    try testing.expectEqual(@as(usize, 0), try shortestPath01(alloc, &adj, 2, 2));
    try testing.expectError(error.NoPath, shortestPath01(alloc, &adj, 0, 2));
}

test "0-1 bfs: invalid input scenarios" {
    const alloc = testing.allocator;
    const bad_weight_adj = [_][]const Edge{
        &[_]Edge{.{ .to = 1, .weight = 2 }},
        &[_]Edge{},
    };
    try testing.expectError(error.InvalidWeight, shortestPath01(alloc, &bad_weight_adj, 0, 1));
    try testing.expectError(error.InvalidNode, shortestPath01(alloc, &bad_weight_adj, 0, 3));
}

test "0-1 bfs: extreme long chain alternating weights" {
    const alloc = testing.allocator;
    const n: usize = 512;

    const mutable_adj = try alloc.alloc([]Edge, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }

    for (0..n) |i| {
        if (i + 1 < n) {
            mutable_adj[i] = try alloc.alloc(Edge, 1);
            mutable_adj[i][0] = .{
                .to = i + 1,
                .weight = @as(u8, @intCast(i % 2)),
            };
        } else {
            mutable_adj[i] = try alloc.alloc(Edge, 0);
        }
    }

    const adj = try alloc.alloc([]const Edge, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    try testing.expectEqual(@as(usize, 255), try shortestPath01(alloc, adj, 0, n - 1));
}
