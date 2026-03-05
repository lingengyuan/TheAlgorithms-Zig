//! Graph Coloring (M-Coloring Problem) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/coloring.py

const std = @import("std");
const testing = std.testing;

pub const ColoringError = error{InvalidGraph};

/// Checks if `color` can be assigned given adjacency row and current assignments.
pub fn validColoring(neighbours: []const u8, colored_vertices: []const i32, target_color: i32) bool {
    for (neighbours, 0..) |neighbour, i| {
        if (neighbour == 1 and colored_vertices[i] == target_color) return false;
    }
    return true;
}

fn utilColor(graph: []const []const u8, max_colors: usize, colored_vertices: []i32, index: usize) bool {
    if (index == graph.len) return true;

    var i: usize = 0;
    while (i < max_colors) : (i += 1) {
        const c: i32 = @intCast(i);
        if (validColoring(graph[index], colored_vertices, c)) {
            colored_vertices[index] = c;
            if (utilColor(graph, max_colors, colored_vertices, index + 1)) return true;
            colored_vertices[index] = -1;
        }
    }
    return false;
}

/// Colors a graph with at most `max_colors` colors.
/// Returns an empty slice if no valid coloring exists.
///
/// Time complexity: O(m^n) worst case
/// Space complexity: O(n)
pub fn color(
    allocator: std.mem.Allocator,
    graph: []const []const u8,
    max_colors: usize,
) (ColoringError || std.mem.Allocator.Error)![]i32 {
    const n = graph.len;
    for (graph) |row| {
        if (row.len != n) return ColoringError.InvalidGraph;
        for (row) |value| {
            if (value != 0 and value != 1) return ColoringError.InvalidGraph;
        }
    }

    const colored_vertices = try allocator.alloc(i32, n);
    defer allocator.free(colored_vertices);
    @memset(colored_vertices, -1);

    if (utilColor(graph, max_colors, colored_vertices, 0)) {
        return allocator.dupe(i32, colored_vertices);
    }
    return allocator.alloc(i32, 0);
}

test "graph coloring: python examples" {
    const alloc = testing.allocator;
    const graph = [_][]const u8{
        &[_]u8{ 0, 1, 0, 0, 0 },
        &[_]u8{ 1, 0, 1, 0, 1 },
        &[_]u8{ 0, 1, 0, 1, 0 },
        &[_]u8{ 0, 1, 1, 0, 0 },
        &[_]u8{ 0, 1, 0, 0, 0 },
    };

    const c3 = try color(alloc, &graph, 3);
    defer alloc.free(c3);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 0, 2, 0 }, c3);

    const c2 = try color(alloc, &graph, 2);
    defer alloc.free(c2);
    try testing.expectEqual(@as(usize, 0), c2.len);
}

test "graph coloring: validation and boundary" {
    const alloc = testing.allocator;
    const bad = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{1},
    };
    try testing.expectError(ColoringError.InvalidGraph, color(alloc, &bad, 2));

    const empty_graph = [_][]const u8{};
    const result = try color(alloc, &empty_graph, 0);
    defer alloc.free(result);
    try testing.expectEqual(@as(usize, 0), result.len);
}

test "graph coloring: extreme complete graph" {
    const alloc = testing.allocator;

    var matrix: [8][8]u8 = undefined;
    for (0..8) |i| {
        for (0..8) |j| {
            matrix[i][j] = if (i == j) 0 else 1;
        }
    }
    const graph = [_][]const u8{
        matrix[0][0..], matrix[1][0..], matrix[2][0..], matrix[3][0..],
        matrix[4][0..], matrix[5][0..], matrix[6][0..], matrix[7][0..],
    };

    const fail = try color(alloc, &graph, 7);
    defer alloc.free(fail);
    try testing.expectEqual(@as(usize, 0), fail.len);

    const ok = try color(alloc, &graph, 8);
    defer alloc.free(ok);
    try testing.expectEqual(@as(usize, 8), ok.len);
}
