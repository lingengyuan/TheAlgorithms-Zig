//! Greedy Best-First Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/greedy_best_first.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Point = struct {
    row: usize,
    col: usize,
};

const Node = struct {
    point: Point,
    parent: ?usize,
    h_cost: usize,
};

/// Greedy best-first search on a grid where 0 = free and 1 = obstacle.
/// Returns path from `start` to `goal` if found.
/// If not found, returns `[start]` to match Python reference behavior.
/// Time complexity: O(V^2) (linear open-set min selection), Space complexity: O(V)
pub fn greedyBestFirstPath(
    allocator: Allocator,
    grid: []const []const u8,
    start: Point,
    goal: Point,
) ![]Point {
    const rows = grid.len;
    if (rows == 0) return error.InvalidGrid;
    const cols = grid[0].len;
    if (cols == 0) return error.InvalidGrid;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    if (start.row >= rows or start.col >= cols) return error.InvalidPoint;
    if (goal.row >= rows or goal.col >= cols) return error.InvalidPoint;

    if (start.row == goal.row and start.col == goal.col) {
        const single = try allocator.alloc(Point, 1);
        single[0] = start;
        return single;
    }

    const cells_mul = @mulWithOverflow(rows, cols);
    if (cells_mul[1] != 0) return error.Overflow;
    const cell_count = cells_mul[0];

    const closed = try allocator.alloc(bool, cell_count);
    defer allocator.free(closed);
    @memset(closed, false);

    var nodes = std.ArrayListUnmanaged(Node){};
    defer nodes.deinit(allocator);
    var open = std.ArrayListUnmanaged(usize){};
    defer open.deinit(allocator);

    try nodes.append(allocator, .{
        .point = start,
        .parent = null,
        .h_cost = manhattan(start, goal),
    });
    try open.append(allocator, 0);

    while (open.items.len > 0) {
        const min_pos = pickMinOpen(nodes.items, open.items);
        const current_idx = open.items[min_pos];
        _ = open.orderedRemove(min_pos);

        const current = nodes.items[current_idx];
        const current_cell = current.point.row * cols + current.point.col;
        if (closed[current_cell]) continue;
        closed[current_cell] = true;

        if (current.point.row == goal.row and current.point.col == goal.col) {
            return try reconstructPath(allocator, nodes.items, current_idx);
        }

        const deltas = [_][2]isize{
            .{ -1, 0 }, // up
            .{ 0, -1 }, // left
            .{ 1, 0 }, // down
            .{ 0, 1 }, // right
        };

        for (deltas) |d| {
            const nr_i = @as(isize, @intCast(current.point.row)) + d[0];
            const nc_i = @as(isize, @intCast(current.point.col)) + d[1];
            if (nr_i < 0 or nc_i < 0) continue;
            const nr = @as(usize, @intCast(nr_i));
            const nc = @as(usize, @intCast(nc_i));
            if (nr >= rows or nc >= cols) continue;
            if (grid[nr][nc] != 0) continue;

            const next_cell = nr * cols + nc;
            if (closed[next_cell]) continue;
            if (isPointInOpen(nodes.items, open.items, .{ .row = nr, .col = nc })) continue;

            try nodes.append(allocator, .{
                .point = .{ .row = nr, .col = nc },
                .parent = current_idx,
                .h_cost = manhattan(.{ .row = nr, .col = nc }, goal),
            });
            try open.append(allocator, nodes.items.len - 1);
        }
    }

    const fallback = try allocator.alloc(Point, 1);
    fallback[0] = start;
    return fallback;
}

fn manhattan(a: Point, b: Point) usize {
    const dr = if (a.row >= b.row) a.row - b.row else b.row - a.row;
    const dc = if (a.col >= b.col) a.col - b.col else b.col - a.col;
    return dr + dc;
}

fn pickMinOpen(nodes: []const Node, open_indices: []const usize) usize {
    var best_pos: usize = 0;
    var best_cost = nodes[open_indices[0]].h_cost;
    for (open_indices, 0..) |node_idx, i| {
        const c = nodes[node_idx].h_cost;
        if (c < best_cost) {
            best_cost = c;
            best_pos = i;
        }
    }
    return best_pos;
}

fn isPointInOpen(nodes: []const Node, open_indices: []const usize, point: Point) bool {
    for (open_indices) |node_idx| {
        const n = nodes[node_idx];
        if (n.point.row == point.row and n.point.col == point.col) return true;
    }
    return false;
}

fn reconstructPath(allocator: Allocator, nodes: []const Node, end_idx: usize) ![]Point {
    var reversed = std.ArrayListUnmanaged(Point){};
    defer reversed.deinit(allocator);

    var cur: ?usize = end_idx;
    while (cur) |idx| {
        try reversed.append(allocator, nodes[idx].point);
        cur = nodes[idx].parent;
    }
    std.mem.reverse(Point, reversed.items);
    return try reversed.toOwnedSlice(allocator);
}

test "greedy best first: python sample path on third grid" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 0, 1, 0, 0 },
        &[_]u8{ 0, 1, 0, 0, 0 },
        &[_]u8{ 0, 0, 1, 0, 1 },
        &[_]u8{ 1, 0, 0, 1, 1 },
        &[_]u8{ 0, 0, 0, 0, 0 },
    };

    const path = try greedyBestFirstPath(alloc, &grid, .{ .row = 0, .col = 0 }, .{ .row = 4, .col = 4 });
    defer alloc.free(path);

    const expected = [_]Point{
        .{ .row = 0, .col = 0 },
        .{ .row = 1, .col = 0 },
        .{ .row = 2, .col = 0 },
        .{ .row = 2, .col = 1 },
        .{ .row = 3, .col = 1 },
        .{ .row = 4, .col = 1 },
        .{ .row = 4, .col = 2 },
        .{ .row = 4, .col = 3 },
        .{ .row = 4, .col = 4 },
    };
    try testing.expectEqual(expected.len, path.len);
    for (expected, 0..) |e, i| {
        try testing.expectEqual(e.row, path[i].row);
        try testing.expectEqual(e.col, path[i].col);
    }
}

test "greedy best first: start equals goal and no path fallback" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{ 1, 1 },
    };

    const same = try greedyBestFirstPath(alloc, &grid, .{ .row = 0, .col = 0 }, .{ .row = 0, .col = 0 });
    defer alloc.free(same);
    try testing.expectEqual(@as(usize, 1), same.len);

    const no_path = try greedyBestFirstPath(alloc, &grid, .{ .row = 0, .col = 0 }, .{ .row = 1, .col = 1 });
    defer alloc.free(no_path);
    try testing.expectEqual(@as(usize, 1), no_path.len);
    try testing.expectEqual(@as(usize, 0), no_path[0].row);
    try testing.expectEqual(@as(usize, 0), no_path[0].col);
}

test "greedy best first: invalid inputs" {
    const alloc = testing.allocator;
    const ragged = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{0},
    };
    try testing.expectError(
        error.InvalidGrid,
        greedyBestFirstPath(alloc, &ragged, .{ .row = 0, .col = 0 }, .{ .row = 1, .col = 0 }),
    );

    const grid = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 0 },
    };
    try testing.expectError(
        error.InvalidPoint,
        greedyBestFirstPath(alloc, &grid, .{ .row = 2, .col = 0 }, .{ .row = 1, .col = 1 }),
    );
}

test "greedy best first: extreme open grid reaches goal" {
    const alloc = testing.allocator;
    const rows: usize = 50;
    const cols: usize = 50;

    const mutable_grid = try alloc.alloc([]u8, rows);
    defer {
        for (mutable_grid) |row| alloc.free(row);
        alloc.free(mutable_grid);
    }
    for (0..rows) |r| {
        mutable_grid[r] = try alloc.alloc(u8, cols);
        @memset(mutable_grid[r], 0);
    }

    const grid = try alloc.alloc([]const u8, rows);
    defer alloc.free(grid);
    for (mutable_grid, 0..) |row, i| grid[i] = row;

    const path = try greedyBestFirstPath(alloc, grid, .{ .row = 0, .col = 0 }, .{ .row = rows - 1, .col = cols - 1 });
    defer alloc.free(path);

    try testing.expect(path.len > 0);
    try testing.expectEqual(@as(usize, 0), path[0].row);
    try testing.expectEqual(@as(usize, 0), path[0].col);
    try testing.expectEqual(rows - 1, path[path.len - 1].row);
    try testing.expectEqual(cols - 1, path[path.len - 1].col);
}
