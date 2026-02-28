//! Cycle Detection in Directed Graph - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/check_cycle.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

fn dfsHasCycle(node: usize, adj: []const []const usize, state: []u8) bool {
    // 0 = unvisited, 1 = in recursion stack, 2 = done
    if (state[node] == 1) return true;
    if (state[node] == 2) return false;

    state[node] = 1;
    const n = adj.len;
    for (adj[node]) |nb| {
        if (nb >= n) continue;
        if (dfsHasCycle(nb, adj, state)) return true;
    }
    state[node] = 2;
    return false;
}

/// Returns true if a directed graph contains a cycle.
/// Graph is adjacency-list based with node ids in [0, n).
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn hasCycle(allocator: Allocator, adj: []const []const usize) !bool {
    const n = adj.len;
    if (n == 0) return false;

    const state = try allocator.alloc(u8, n);
    defer allocator.free(state);
    @memset(state, 0);

    for (0..n) |i| {
        if (state[i] == 0 and dfsHasCycle(i, adj, state)) {
            return true;
        }
    }
    return false;
}

test "detect cycle: acyclic graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{3},
        &[_]usize{3},
        &[_]usize{},
    };
    try testing.expect(!(try hasCycle(alloc, &adj)));
}

test "detect cycle: simple cycle" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
        &[_]usize{0},
    };
    try testing.expect(try hasCycle(alloc, &adj));
}

test "detect cycle: self loop" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{0},
    };
    try testing.expect(try hasCycle(alloc, &adj));
}

test "detect cycle: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 9 },
        &[_]usize{},
    };
    try testing.expect(!(try hasCycle(alloc, &adj)));
}

test "detect cycle: empty graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};
    try testing.expect(!(try hasCycle(alloc, &adj)));
}
