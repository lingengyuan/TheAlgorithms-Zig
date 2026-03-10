//! Project Euler Problem 107: Minimal Network - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_107/sol1.py

const std = @import("std");
const testing = std.testing;

const network_file = @embedFile("problem_107_network.txt");
const network_test_file = @embedFile("problem_107_test_network.txt");

pub const Problem107Error = error{ InvalidMatrix, InvalidNumber, Disconnected };

const no_edge = std.math.maxInt(u32);

/// Returns the maximum saving after replacing the network with its minimum spanning tree.
/// Time complexity: O(n^3)
/// Space complexity: O(n^2)
pub fn solution(allocator: std.mem.Allocator, data: []const u8) Problem107Error!u32 {
    var line_counter = std.mem.tokenizeAny(u8, data, "\r\n");
    var size: usize = 0;
    while (line_counter.next()) |_| size += 1;
    if (size == 0) return error.InvalidMatrix;

    const weights = allocator.alloc(u32, size * size) catch return error.InvalidMatrix;
    defer allocator.free(weights);
    @memset(weights, no_edge);

    var row: usize = 0;
    var total_weight: u32 = 0;
    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| : (row += 1) {
        if (row >= size) return error.InvalidMatrix;
        var cols = std.mem.tokenizeScalar(u8, line, ',');
        var col: usize = 0;
        while (cols.next()) |item| : (col += 1) {
            if (col >= size) return error.InvalidMatrix;
            const value = if (item.len == 1 and item[0] == '-') no_edge else std.fmt.parseInt(u32, item, 10) catch return error.InvalidNumber;
            weights[row * size + col] = value;
            if (row > col and value != no_edge) total_weight += value;
        }
        if (col != size) return error.InvalidMatrix;
    }
    if (row != size) return error.InvalidMatrix;

    const in_tree = allocator.alloc(bool, size) catch return error.InvalidMatrix;
    defer allocator.free(in_tree);
    @memset(in_tree, false);
    in_tree[0] = true;

    var tree_weight: u32 = 0;
    var tree_size: usize = 1;
    while (tree_size < size) : (tree_size += 1) {
        var best_weight: u32 = no_edge;
        var best_vertex: ?usize = null;

        for (0..size) |from| {
            if (!in_tree[from]) continue;
            for (0..size) |to| {
                if (in_tree[to]) continue;
                const weight = weights[from * size + to];
                if (weight < best_weight) {
                    best_weight = weight;
                    best_vertex = to;
                }
            }
        }

        const next_vertex = best_vertex orelse return error.Disconnected;
        in_tree[next_vertex] = true;
        tree_weight += best_weight;
    }

    return total_weight - tree_weight;
}

test "problem 107: python reference" {
    try testing.expectEqual(@as(u32, 259679), try solution(testing.allocator, network_file));
}

test "problem 107: sample network" {
    try testing.expectEqual(@as(u32, 150), try solution(testing.allocator, network_test_file));
}

test "problem 107: invalid and disconnected inputs" {
    try testing.expectError(error.InvalidMatrix, solution(testing.allocator, "-,1\n2\n"));
    try testing.expectError(error.Disconnected, solution(testing.allocator, "-,-\n-,-\n"));
}
