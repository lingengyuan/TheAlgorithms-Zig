//! Floyd-Warshall (DP graph wrapper) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/floyd_warshall.py

const std = @import("std");
const testing = std.testing;
pub const GraphError = error{ Overflow, InvalidNode, InvalidMatrixSize };
const inf: i64 = 1_000_000_000;

pub const Graph = struct {
    allocator: std.mem.Allocator,
    n: usize,
    dp: []i64,

    pub fn init(allocator: std.mem.Allocator, n: usize) !Graph {
        const elem_count = try std.math.mul(usize, n, n);
        const dp = try allocator.alloc(i64, elem_count);
        for (0..n) |i| {
            for (0..n) |j| {
                dp[i * n + j] = if (i == j) 0 else inf;
            }
        }
        return .{ .allocator = allocator, .n = n, .dp = dp };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.dp);
    }

    pub fn addEdge(self: *Graph, u: usize, v: usize, w: i64) GraphError!void {
        if (u >= self.n or v >= self.n) return error.InvalidNode;
        self.dp[u * self.n + v] = w;
    }

    pub fn floydWarshall(self: *Graph) GraphError!void {
        if (self.n == 0) return;
        for (0..self.n) |k| {
            for (0..self.n) |i| {
                const ik = self.dp[i * self.n + k];
                if (ik == inf) continue;
                for (0..self.n) |j| {
                    const kj = self.dp[k * self.n + j];
                    if (kj == inf) continue;
                    const sum = @addWithOverflow(ik, kj);
                    if (sum[1] != 0) return error.Overflow;
                    const index = i * self.n + j;
                    if (sum[0] < self.dp[index]) self.dp[index] = sum[0];
                }
            }
        }
    }

    /// Returns `null` for unreachable pairs instead of Python's `math.inf`.
    pub fn showMin(self: Graph, u: usize, v: usize) GraphError!?i64 {
        if (u >= self.n or v >= self.n) return error.InvalidNode;
        const value = self.dp[u * self.n + v];
        return if (value == inf) null else value;
    }
};

test "dynamic programming floyd warshall: python examples" {
    var g = try Graph.init(testing.allocator, 3);
    defer g.deinit();
    try g.addEdge(0, 1, 1);
    try g.addEdge(1, 2, 2);
    try g.floydWarshall();
    try testing.expectEqual(@as(?i64, 3), try g.showMin(0, 2));
    try testing.expectEqual(@as(?i64, null), try g.showMin(2, 0));
}

test "dynamic programming floyd warshall: another route" {
    var g = try Graph.init(testing.allocator, 3);
    defer g.deinit();
    try g.addEdge(0, 1, 3);
    try g.addEdge(1, 2, 4);
    try g.floydWarshall();
    try testing.expectEqual(@as(?i64, 7), try g.showMin(0, 2));
    try testing.expectEqual(@as(?i64, null), try g.showMin(1, 0));
}

test "dynamic programming floyd warshall: invalid node and empty graph" {
    var g = try Graph.init(testing.allocator, 0);
    defer g.deinit();
    try g.floydWarshall();
    try testing.expectError(error.InvalidNode, g.addEdge(0, 0, 1));
    try testing.expectError(error.InvalidNode, g.showMin(0, 0));
}
