//! Hamiltonian Cycle - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/hamiltonian_cycle.py

const std = @import("std");
const testing = std.testing;

pub const HamiltonianError = error{ InvalidGraph, InvalidStartIndex };

fn validConnection(graph: []const []const u8, next_ver: usize, curr_ind: usize, path: []isize) bool {
    const prev: usize = @intCast(path[curr_ind - 1]);
    if (graph[prev][next_ver] == 0) return false;

    for (path) |vertex| {
        if (vertex == @as(isize, @intCast(next_ver))) return false;
    }
    return true;
}

fn utilHamiltonCycle(graph: []const []const u8, path: []isize, curr_ind: usize) bool {
    if (curr_ind == graph.len) {
        const prev: usize = @intCast(path[curr_ind - 1]);
        const start: usize = @intCast(path[0]);
        return graph[prev][start] == 1;
    }

    var next_ver: usize = 0;
    while (next_ver < graph.len) : (next_ver += 1) {
        if (validConnection(graph, next_ver, curr_ind, path)) {
            path[curr_ind] = @intCast(next_ver);
            if (utilHamiltonCycle(graph, path, curr_ind + 1)) return true;
            path[curr_ind] = -1;
        }
    }

    return false;
}

/// Returns Hamiltonian cycle path (start repeated at end) or empty slice when absent.
///
/// Time complexity: O(n!)
/// Space complexity: O(n)
pub fn hamiltonCycle(
    allocator: std.mem.Allocator,
    graph: []const []const u8,
    start_index: usize,
) (HamiltonianError || std.mem.Allocator.Error)![]usize {
    const n = graph.len;
    if (n == 0) return HamiltonianError.InvalidGraph;

    for (graph) |row| {
        if (row.len != n) return HamiltonianError.InvalidGraph;
        for (row) |value| {
            if (value != 0 and value != 1) return HamiltonianError.InvalidGraph;
        }
    }

    if (start_index >= n) return HamiltonianError.InvalidStartIndex;

    var path = try allocator.alloc(isize, n + 1);
    defer allocator.free(path);

    @memset(path, -1);
    path[0] = @intCast(start_index);
    path[n] = @intCast(start_index);

    if (!utilHamiltonCycle(graph, path, 1)) {
        return allocator.alloc(usize, 0);
    }

    const out = try allocator.alloc(usize, n + 1);
    for (0..n + 1) |i| {
        out[i] = @intCast(path[i]);
    }
    return out;
}

test "hamiltonian cycle: python case 1" {
    const alloc = testing.allocator;
    const graph = [_][]const u8{
        &[_]u8{ 0, 1, 0, 1, 0 },
        &[_]u8{ 1, 0, 1, 1, 1 },
        &[_]u8{ 0, 1, 0, 0, 1 },
        &[_]u8{ 1, 1, 0, 0, 1 },
        &[_]u8{ 0, 1, 1, 1, 0 },
    };

    const path = try hamiltonCycle(alloc, &graph, 0);
    defer alloc.free(path);

    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 4, 3, 0 }, path);
}

test "hamiltonian cycle: python case 2 and case 3" {
    const alloc = testing.allocator;

    const graph1 = [_][]const u8{
        &[_]u8{ 0, 1, 0, 1, 0 },
        &[_]u8{ 1, 0, 1, 1, 1 },
        &[_]u8{ 0, 1, 0, 0, 1 },
        &[_]u8{ 1, 1, 0, 0, 1 },
        &[_]u8{ 0, 1, 1, 1, 0 },
    };
    const path1 = try hamiltonCycle(alloc, &graph1, 3);
    defer alloc.free(path1);
    try testing.expectEqualSlices(usize, &[_]usize{ 3, 0, 1, 2, 4, 3 }, path1);

    const graph2 = [_][]const u8{
        &[_]u8{ 0, 1, 0, 1, 0 },
        &[_]u8{ 1, 0, 1, 1, 1 },
        &[_]u8{ 0, 1, 0, 0, 1 },
        &[_]u8{ 1, 1, 0, 0, 0 },
        &[_]u8{ 0, 1, 1, 0, 0 },
    };
    const path2 = try hamiltonCycle(alloc, &graph2, 4);
    defer alloc.free(path2);
    try testing.expectEqual(@as(usize, 0), path2.len);
}

test "hamiltonian cycle: validation" {
    const alloc = testing.allocator;

    const bad_graph = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{1},
    };
    try testing.expectError(HamiltonianError.InvalidGraph, hamiltonCycle(alloc, &bad_graph, 0));

    const graph = [_][]const u8{&[_]u8{0}};
    try testing.expectError(HamiltonianError.InvalidStartIndex, hamiltonCycle(alloc, &graph, 1));
}
