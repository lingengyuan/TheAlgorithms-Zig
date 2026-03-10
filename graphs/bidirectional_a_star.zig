//! Bidirectional A* Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/bidirectional_a_star.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Position = struct {
    row: usize,
    col: usize,
};

const Node = struct {
    pos: Position,
    goal: Position,
    g_cost: usize,
    parent: ?Position,

    fn hCost(self: Node, heuristic: Heuristic) f64 {
        const dx = @as(f64, @floatFromInt(self.pos.col)) - @as(f64, @floatFromInt(self.goal.col));
        const dy = @as(f64, @floatFromInt(self.pos.row)) - @as(f64, @floatFromInt(self.goal.row));
        return switch (heuristic) {
            .euclidean => @sqrt(dx * dx + dy * dy),
            .manhattan => @abs(dx) + @abs(dy),
        };
    }

    fn fCost(self: Node, heuristic: Heuristic) f64 {
        return @as(f64, @floatFromInt(self.g_cost)) + self.hCost(heuristic);
    }
};

pub const Heuristic = enum {
    euclidean,
    manhattan,
};

const Frontier = struct {
    allocator: Allocator,
    open_nodes: std.ArrayListUnmanaged(Node),
    closed_nodes: std.ArrayListUnmanaged(Node),
    target: Position,

    fn init(allocator: Allocator, start: Position, goal: Position) !Frontier {
        var open_nodes = std.ArrayListUnmanaged(Node){};
        try open_nodes.append(allocator, .{
            .pos = start,
            .goal = goal,
            .g_cost = 0,
            .parent = null,
        });

        return .{
            .allocator = allocator,
            .open_nodes = open_nodes,
            .closed_nodes = .{},
            .target = goal,
        };
    }

    fn deinit(self: *Frontier) void {
        self.open_nodes.deinit(self.allocator);
        self.closed_nodes.deinit(self.allocator);
    }

    fn retracePath(self: *const Frontier, allocator: Allocator, node: Node) ![]Position {
        var reversed = std.ArrayListUnmanaged(Position){};
        defer reversed.deinit(allocator);

        var current = node;
        while (true) {
            try reversed.append(allocator, current.pos);
            const parent = current.parent orelse break;
            current = self.findClosed(parent) orelse self.findOpen(parent) orelse return error.InternalInvariantBroken;
        }

        std.mem.reverse(Position, reversed.items);
        return try reversed.toOwnedSlice(allocator);
    }

    fn findClosed(self: *const Frontier, pos: Position) ?Node {
        for (self.closed_nodes.items) |node| {
            if (eqlPos(node.pos, pos)) return node;
        }
        return null;
    }

    fn findOpen(self: *const Frontier, pos: Position) ?Node {
        for (self.open_nodes.items) |node| {
            if (eqlPos(node.pos, pos)) return node;
        }
        return null;
    }
};

/// Performs bidirectional A* on a binary grid where `0` is free space and non-zero values are blocked.
/// Returns a path from `start` to `goal`, or a single-node path containing `start` if no path exists.
/// On symmetric shortest-path grids, tie-breaking may produce a different but equally short
/// valid path than the Python sample output.
/// Time complexity: O(R * C * frontier_sort), Space complexity: O(R * C)
pub fn bidirectionalAStar(
    allocator: Allocator,
    grid: []const []const u8,
    start: Position,
    goal: Position,
    heuristic: Heuristic,
) ![]Position {
    try validateGrid(grid, start, goal);
    if (eqlPos(start, goal)) {
        const out = try allocator.alloc(Position, 1);
        out[0] = start;
        return out;
    }

    var forward = try Frontier.init(allocator, start, goal);
    defer forward.deinit();
    var backward = try Frontier.init(allocator, goal, start);
    defer backward.deinit();

    while (forward.open_nodes.items.len > 0 and backward.open_nodes.items.len > 0) {
        sortNodes(forward.open_nodes.items, heuristic);
        sortNodes(backward.open_nodes.items, heuristic);

        const current_fwd = forward.open_nodes.orderedRemove(0);
        const current_bwd = backward.open_nodes.orderedRemove(0);

        if (eqlPos(current_fwd.pos, current_bwd.pos)) {
            return retraceBidirectionalPath(allocator, &forward, &backward, current_fwd, current_bwd);
        }

        try forward.closed_nodes.append(allocator, current_fwd);
        try backward.closed_nodes.append(allocator, current_bwd);

        if (backward.findClosed(current_fwd.pos)) |other| {
            return retraceBidirectionalPath(allocator, &forward, &backward, current_fwd, other);
        }
        if (backward.findOpen(current_fwd.pos)) |other| {
            return retraceBidirectionalPath(allocator, &forward, &backward, current_fwd, other);
        }
        if (forward.findClosed(current_bwd.pos)) |other| {
            return retraceBidirectionalPath(allocator, &forward, &backward, other, current_bwd);
        }
        if (forward.findOpen(current_bwd.pos)) |other| {
            return retraceBidirectionalPath(allocator, &forward, &backward, other, current_bwd);
        }

        forward.target = current_bwd.pos;
        backward.target = current_fwd.pos;

        if (try expandFrontier(allocator, grid, &forward, &backward, current_fwd, heuristic)) |meet| {
            return retraceBidirectionalPath(allocator, &forward, &backward, meet.self_node, meet.other_node);
        }
        if (try expandFrontier(allocator, grid, &backward, &forward, current_bwd, heuristic)) |meet| {
            return retraceBidirectionalPath(allocator, &forward, &backward, meet.other_node, meet.self_node);
        }
    }

    const out = try allocator.alloc(Position, 1);
    out[0] = start;
    return out;
}

fn validateGrid(grid: []const []const u8, start: Position, goal: Position) !void {
    if (grid.len == 0) return error.InvalidGrid;
    const width = grid[0].len;
    if (width == 0) return error.InvalidGrid;
    for (grid) |row| {
        if (row.len != width) return error.InvalidGrid;
    }

    if (start.row >= grid.len or start.col >= width or goal.row >= grid.len or goal.col >= width) {
        return error.InvalidPosition;
    }
    if (grid[start.row][start.col] != 0 or grid[goal.row][goal.col] != 0) return error.BlockedCell;
}

fn expandFrontier(
    allocator: Allocator,
    grid: []const []const u8,
    frontier: *Frontier,
    opposite: *const Frontier,
    current: Node,
    heuristic: Heuristic,
) !?struct { self_node: Node, other_node: Node } {
    const directions = [_][2]isize{
        .{ -1, 0 },
        .{ 0, -1 },
        .{ 1, 0 },
        .{ 0, 1 },
    };

    for (directions) |delta| {
        const next_row_signed = @as(isize, @intCast(current.pos.row)) + delta[0];
        const next_col_signed = @as(isize, @intCast(current.pos.col)) + delta[1];
        if (next_row_signed < 0 or next_col_signed < 0) continue;

        const next = Position{
            .row = @as(usize, @intCast(next_row_signed)),
            .col = @as(usize, @intCast(next_col_signed)),
        };
        if (next.row >= grid.len or next.col >= grid[0].len) continue;
        if (grid[next.row][next.col] != 0) continue;
        if (frontier.findClosed(next) != null) continue;

        const child = Node{
            .pos = next,
            .goal = frontier.target,
            .g_cost = current.g_cost + 1,
            .parent = current.pos,
        };

        if (indexOfPosition(frontier.open_nodes.items, next)) |idx| {
            if (child.g_cost < frontier.open_nodes.items[idx].g_cost) {
                frontier.open_nodes.items[idx] = child;
            }
        } else {
            try frontier.open_nodes.append(allocator, child);
        }

        if (opposite.findClosed(next)) |other| {
            return .{ .self_node = child, .other_node = other };
        }
        if (opposite.findOpen(next)) |other| {
            return .{ .self_node = child, .other_node = other };
        }
    }

    sortNodes(frontier.open_nodes.items, heuristic);
    return null;
}

fn retraceBidirectionalPath(
    allocator: Allocator,
    forward: *const Frontier,
    backward: *const Frontier,
    fwd_node: Node,
    bwd_node: Node,
) ![]Position {
    const fwd_path = try forward.retracePath(allocator, fwd_node);
    defer allocator.free(fwd_path);
    const bwd_path = try backward.retracePath(allocator, bwd_node);
    defer allocator.free(bwd_path);

    if (bwd_path.len == 0) return error.InternalInvariantBroken;

    const out = try allocator.alloc(Position, fwd_path.len + bwd_path.len - 1);
    @memcpy(out[0..fwd_path.len], fwd_path);

    var write_index = fwd_path.len;
    var i = bwd_path.len - 1;
    while (i > 0) : (i -= 1) {
        out[write_index] = bwd_path[i - 1];
        write_index += 1;
    }
    return out;
}

fn sortNodes(nodes: []Node, heuristic: Heuristic) void {
    std.sort.heap(Node, nodes, heuristic, lessNode);
}

fn lessNode(heuristic: Heuristic, a: Node, b: Node) bool {
    const af = a.fCost(heuristic);
    const bf = b.fCost(heuristic);
    if (af != bf) return af < bf;
    return a.g_cost < b.g_cost;
}

fn eqlPos(a: Position, b: Position) bool {
    return a.row == b.row and a.col == b.col;
}

fn indexOfPosition(nodes: []const Node, pos: Position) ?usize {
    for (nodes, 0..) |node, i| {
        if (eqlPos(node.pos, pos)) return i;
    }
    return null;
}

test "bidirectional a star: python sample length and validity" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 1, 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 0, 1, 0, 0, 0, 0 },
        &[_]u8{ 1, 0, 1, 0, 0, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 0, 0, 0, 1, 0, 0 },
    };

    const path = try bidirectionalAStar(alloc, &grid, .{ .row = 0, .col = 0 }, .{ .row = 6, .col = 6 }, .euclidean);
    defer alloc.free(path);

    try testing.expectEqual(@as(usize, 13), path.len);
    try testing.expectEqual(Position{ .row = 0, .col = 0 }, path[0]);
    try testing.expectEqual(Position{ .row = 6, .col = 6 }, path[path.len - 1]);
    for (path[1..], 0..) |pos, i| {
        const prev = path[i];
        const row_delta = @abs(@as(i64, @intCast(pos.row)) - @as(i64, @intCast(prev.row)));
        const col_delta = @abs(@as(i64, @intCast(pos.col)) - @as(i64, @intCast(prev.col)));
        try testing.expectEqual(@as(u64, 1), row_delta + col_delta);
        try testing.expectEqual(@as(u8, 0), grid[pos.row][pos.col]);
    }
}

test "bidirectional a star: start equals goal" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 0 },
    };

    const path = try bidirectionalAStar(alloc, &grid, .{ .row = 1, .col = 1 }, .{ .row = 1, .col = 1 }, .euclidean);
    defer alloc.free(path);
    try testing.expectEqualSlices(Position, &[_]Position{.{ .row = 1, .col = 1 }}, path);
}

test "bidirectional a star: blocked or invalid endpoints" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{ 0, 0 },
    };

    try testing.expectError(error.InvalidPosition, bidirectionalAStar(alloc, &grid, .{ .row = 0, .col = 2 }, .{ .row = 1, .col = 1 }, .euclidean));
    try testing.expectError(error.BlockedCell, bidirectionalAStar(alloc, &grid, .{ .row = 0, .col = 1 }, .{ .row = 1, .col = 1 }, .euclidean));
}

test "bidirectional a star: unreachable returns start only" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 1, 1, 1 },
        &[_]u8{ 0, 1, 0 },
    };

    const path = try bidirectionalAStar(alloc, &grid, .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 2 }, .manhattan);
    defer alloc.free(path);
    try testing.expectEqualSlices(Position, &[_]Position{.{ .row = 0, .col = 0 }}, path);
}

test "bidirectional a star: extreme empty corridor" {
    const alloc = testing.allocator;
    const n: usize = 40;
    const data = try alloc.alloc(u8, n * n);
    defer alloc.free(data);
    @memset(data, 0);

    const rows = try alloc.alloc([]const u8, n);
    defer alloc.free(rows);
    for (0..n) |i| rows[i] = data[i * n .. (i + 1) * n];

    const path = try bidirectionalAStar(alloc, rows, .{ .row = 0, .col = 0 }, .{ .row = n - 1, .col = n - 1 }, .euclidean);
    defer alloc.free(path);
    try testing.expect(path.len >= (2 * n - 1));
    try testing.expectEqual(Position{ .row = 0, .col = 0 }, path[0]);
    try testing.expectEqual(Position{ .row = n - 1, .col = n - 1 }, path[path.len - 1]);
}
