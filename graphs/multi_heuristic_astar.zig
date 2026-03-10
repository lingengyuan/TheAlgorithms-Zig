//! Multi-Heuristic A* - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/multi_heuristic_astar.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Position = struct {
    x: usize,
    y: usize,
};

pub const SearchResult = struct {
    path: []Position,
    cost: usize,

    pub fn deinit(self: SearchResult, allocator: Allocator) void {
        allocator.free(self.path);
    }
};

const HeuristicCount = 3;
const W1: f64 = 1.0;
const W2: f64 = 1.0;

/// Runs a grid-based multi-heuristic A* search using one anchor heuristic
/// (Euclidean distance) plus two inadmissible/alternate heuristics
/// (Manhattan distance and time-scaled Euclidean distance).
/// The Python reference prints the discovered path; this Zig version returns
/// the path and total cost directly for testability.
/// Time complexity: O(H * V² + H * E * V) in this array-backed implementation
/// Space complexity: O(H * V)
pub fn multiHeuristicAStar(
    allocator: Allocator,
    grid: []const []const u8,
    start: Position,
    goal: Position,
) !SearchResult {
    try validateGrid(grid, start, goal);
    if (eql(start, goal)) {
        const path = try allocator.alloc(Position, 1);
        path[0] = start;
        return .{ .path = path, .cost = 0 };
    }

    const rows = grid.len;
    const cols = grid[0].len;
    const count = rows * cols;
    const inf = std.math.inf(f64);
    const none = std.math.maxInt(usize);

    const g_score = try allocator.alloc(f64, count);
    defer allocator.free(g_score);
    const parent = try allocator.alloc(usize, count);
    defer allocator.free(parent);
    const close_anchor = try allocator.alloc(bool, count);
    defer allocator.free(close_anchor);
    const close_inad = try allocator.alloc(bool, count);
    defer allocator.free(close_inad);
    const open_lists = try allocator.alloc([]bool, HeuristicCount);
    defer allocator.free(open_lists);
    const open_storage = try allocator.alloc(bool, HeuristicCount * count);
    defer allocator.free(open_storage);

    @memset(g_score, inf);
    @memset(parent, none);
    @memset(close_anchor, false);
    @memset(close_inad, false);
    @memset(open_storage, false);
    for (0..HeuristicCount) |i| {
        open_lists[i] = open_storage[i * count .. (i + 1) * count];
    }

    const start_idx = toIndex(start, cols);
    const goal_idx = toIndex(goal, cols);
    g_score[start_idx] = 0.0;
    for (open_lists) |open_list| open_list[start_idx] = true;

    var time_scale: f64 = 1.0;
    while (true) {
        const anchor_min = minOpenKey(open_lists[0], g_score, grid, cols, goal, 0, time_scale) orelse break;

        var expanded_inad = false;
        for (1..HeuristicCount) |heuristic_idx| {
            const inad_min = minOpenKey(open_lists[heuristic_idx], g_score, grid, cols, goal, heuristic_idx, time_scale) orelse continue;
            if (inad_min.key <= W2 * anchor_min.key) {
                time_scale += 1.0;
                if (g_score[goal_idx] <= inad_min.key) {
                    if (g_score[goal_idx] != inf) {
                        return .{
                            .path = try reconstructPath(allocator, parent, cols, start_idx, goal_idx),
                            .cost = @as(usize, @intFromFloat(g_score[goal_idx])),
                        };
                    }
                } else {
                    try expandState(
                        allocator,
                        grid,
                        inad_min.idx,
                        goal,
                        cols,
                        heuristic_idx,
                        time_scale,
                        g_score,
                        parent,
                        close_anchor,
                        close_inad,
                        open_lists,
                    );
                    close_inad[inad_min.idx] = true;
                }
                expanded_inad = true;
                break;
            }
        }

        if (expanded_inad) continue;

        if (g_score[goal_idx] <= anchor_min.key) {
            if (g_score[goal_idx] != inf) {
                return .{
                    .path = try reconstructPath(allocator, parent, cols, start_idx, goal_idx),
                    .cost = @as(usize, @intFromFloat(g_score[goal_idx])),
                };
            }
        } else {
            try expandState(
                allocator,
                grid,
                anchor_min.idx,
                goal,
                cols,
                0,
                time_scale,
                g_score,
                parent,
                close_anchor,
                close_inad,
                open_lists,
            );
            close_anchor[anchor_min.idx] = true;
        }
    }

    return error.NoPath;
}

fn validateGrid(grid: []const []const u8, start: Position, goal: Position) !void {
    if (grid.len == 0) return error.InvalidGrid;
    const cols = grid[0].len;
    if (cols == 0) return error.InvalidGrid;
    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }
    if (start.y >= grid.len or goal.y >= grid.len or start.x >= cols or goal.x >= cols) {
        return error.InvalidPosition;
    }
    if (grid[start.y][start.x] != 0 or grid[goal.y][goal.x] != 0) return error.BlockedCell;
}

const MinEntry = struct {
    idx: usize,
    key: f64,
    g: f64,
};

fn minOpenKey(
    open_list: []const bool,
    g_score: []const f64,
    grid: []const []const u8,
    cols: usize,
    goal: Position,
    heuristic_idx: usize,
    time_scale: f64,
) ?MinEntry {
    var best: ?MinEntry = null;
    for (open_list, 0..) |present, idx| {
        if (!present) continue;
        const pos = fromIndex(idx, cols);
        if (grid[pos.y][pos.x] != 0) continue;
        const g = g_score[idx];
        if (!std.math.isFinite(g)) continue;
        const key_value = keyFor(pos, goal, g, heuristic_idx, time_scale);
        if (best == null or key_value < best.?.key or (key_value == best.?.key and g < best.?.g)) {
            best = .{ .idx = idx, .key = key_value, .g = g };
        }
    }
    return best;
}

fn keyFor(pos: Position, goal: Position, g_value: f64, heuristic_idx: usize, time_scale: f64) f64 {
    return g_value + W1 * heuristic(pos, goal, heuristic_idx, time_scale);
}

fn heuristic(pos: Position, goal: Position, heuristic_idx: usize, time_scale: f64) f64 {
    const dx = @as(f64, @floatFromInt(if (pos.x >= goal.x) pos.x - goal.x else goal.x - pos.x));
    const dy = @as(f64, @floatFromInt(if (pos.y >= goal.y) pos.y - goal.y else goal.y - pos.y));
    const euclidean = @sqrt(dx * dx + dy * dy);
    return switch (heuristic_idx) {
        0 => euclidean,
        1 => dx + dy,
        2 => euclidean / @max(time_scale, 1.0),
        else => unreachable,
    };
}

fn expandState(
    allocator: Allocator,
    grid: []const []const u8,
    state_idx: usize,
    goal: Position,
    cols: usize,
    current_queue: usize,
    time_scale: f64,
    g_score: []f64,
    parent: []usize,
    close_anchor: []const bool,
    close_inad: []const bool,
    open_lists: [][]bool,
) !void {
    _ = allocator;
    for (open_lists) |open_list| {
        open_list[state_idx] = false;
    }

    const current = fromIndex(state_idx, cols);
    const directions = [_][2]isize{
        .{ -1, 0 },
        .{ 1, 0 },
        .{ 0, -1 },
        .{ 0, 1 },
    };

    for (directions) |delta| {
        const next_x_signed = @as(isize, @intCast(current.x)) + delta[0];
        const next_y_signed = @as(isize, @intCast(current.y)) + delta[1];
        if (next_x_signed < 0 or next_y_signed < 0) continue;

        const next = Position{
            .x = @as(usize, @intCast(next_x_signed)),
            .y = @as(usize, @intCast(next_y_signed)),
        };
        if (next.y >= grid.len or next.x >= cols) continue;
        if (grid[next.y][next.x] != 0) continue;

        const next_idx = toIndex(next, cols);
        const tentative_g = g_score[state_idx] + 1.0;
        if (tentative_g < g_score[next_idx]) {
            g_score[next_idx] = tentative_g;
            parent[next_idx] = state_idx;
            if (!close_anchor[next_idx]) {
                open_lists[0][next_idx] = true;
                if (!close_inad[next_idx]) {
                    const anchor_key = keyFor(next, goal, tentative_g, 0, time_scale);
                    for (1..HeuristicCount) |heuristic_idx| {
                        if (keyFor(next, goal, tentative_g, heuristic_idx, time_scale) <= W2 * anchor_key) {
                            open_lists[heuristic_idx][next_idx] = true;
                        }
                    }
                    if (current_queue > 0) {
                        open_lists[current_queue][next_idx] = true;
                    }
                }
            }
        }
    }
}

fn reconstructPath(
    allocator: Allocator,
    parent: []const usize,
    cols: usize,
    start_idx: usize,
    goal_idx: usize,
) ![]Position {
    const none = std.math.maxInt(usize);
    var len: usize = 1;
    var cursor = goal_idx;
    while (cursor != start_idx) {
        cursor = parent[cursor];
        if (cursor == none) return error.NoPath;
        len += 1;
    }

    const path = try allocator.alloc(Position, len);
    var write_idx = len;
    cursor = goal_idx;
    while (true) {
        write_idx -= 1;
        path[write_idx] = fromIndex(cursor, cols);
        if (cursor == start_idx) break;
        cursor = parent[cursor];
    }
    return path;
}

fn eql(a: Position, b: Position) bool {
    return a.x == b.x and a.y == b.y;
}

fn toIndex(pos: Position, cols: usize) usize {
    return pos.y * cols + pos.x;
}

fn fromIndex(idx: usize, cols: usize) Position {
    return .{ .x = idx % cols, .y = idx / cols };
}

fn shortestPathLengthBfs(allocator: Allocator, grid: []const []const u8, start: Position, goal: Position) !?usize {
    const cols = grid[0].len;
    const count = grid.len * cols;
    const visited = try allocator.alloc(bool, count);
    defer allocator.free(visited);
    @memset(visited, false);

    var queue = std.ArrayListUnmanaged(struct { pos: Position, dist: usize }){};
    defer queue.deinit(allocator);
    try queue.append(allocator, .{ .pos = start, .dist = 0 });
    visited[toIndex(start, cols)] = true;

    var head: usize = 0;
    while (head < queue.items.len) : (head += 1) {
        const item = queue.items[head];
        if (eql(item.pos, goal)) return item.dist;
        const directions = [_][2]isize{
            .{ -1, 0 },
            .{ 1, 0 },
            .{ 0, -1 },
            .{ 0, 1 },
        };
        for (directions) |delta| {
            const next_x_signed = @as(isize, @intCast(item.pos.x)) + delta[0];
            const next_y_signed = @as(isize, @intCast(item.pos.y)) + delta[1];
            if (next_x_signed < 0 or next_y_signed < 0) continue;
            const next = Position{
                .x = @as(usize, @intCast(next_x_signed)),
                .y = @as(usize, @intCast(next_y_signed)),
            };
            if (next.y >= grid.len or next.x >= cols) continue;
            if (grid[next.y][next.x] != 0) continue;
            const idx = toIndex(next, cols);
            if (visited[idx]) continue;
            visited[idx] = true;
            try queue.append(allocator, .{ .pos = next, .dist = item.dist + 1 });
        }
    }

    return null;
}

test "multi heuristic astar: basic shortest path agrees with bfs" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 0, 0, 0, 0 },
        &[_]u8{ 0, 1, 1, 1, 0 },
        &[_]u8{ 0, 0, 0, 1, 0 },
        &[_]u8{ 0, 1, 0, 0, 0 },
        &[_]u8{ 0, 0, 0, 1, 0 },
    };

    var result = try multiHeuristicAStar(alloc, &grid, .{ .x = 0, .y = 0 }, .{ .x = 4, .y = 4 });
    defer result.deinit(alloc);

    const expected = (try shortestPathLengthBfs(alloc, &grid, .{ .x = 0, .y = 0 }, .{ .x = 4, .y = 4 })).?;
    try testing.expectEqual(expected, result.cost);
    try testing.expectEqual(result.cost + 1, result.path.len);
    try testing.expectEqual(Position{ .x = 0, .y = 0 }, result.path[0]);
    try testing.expectEqual(Position{ .x = 4, .y = 4 }, result.path[result.path.len - 1]);
}

test "multi heuristic astar: start equals goal" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 0 },
        &[_]u8{ 0, 0 },
    };

    var result = try multiHeuristicAStar(alloc, &grid, .{ .x = 1, .y = 1 }, .{ .x = 1, .y = 1 });
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 0), result.cost);
    try testing.expectEqualSlices(Position, &[_]Position{.{ .x = 1, .y = 1 }}, result.path);
}

test "multi heuristic astar: unreachable grid returns error" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 1, 1, 1 },
        &[_]u8{ 0, 1, 0 },
    };

    try testing.expectError(error.NoPath, multiHeuristicAStar(alloc, &grid, .{ .x = 0, .y = 0 }, .{ .x = 2, .y = 2 }));
}

test "multi heuristic astar: invalid and blocked endpoints" {
    const alloc = testing.allocator;
    const grid = [_][]const u8{
        &[_]u8{ 0, 1 },
        &[_]u8{ 0, 0 },
    };

    try testing.expectError(error.BlockedCell, multiHeuristicAStar(alloc, &grid, .{ .x = 0, .y = 0 }, .{ .x = 1, .y = 0 }));
    try testing.expectError(error.InvalidPosition, multiHeuristicAStar(alloc, &grid, .{ .x = 2, .y = 0 }, .{ .x = 1, .y = 1 }));
}

test "multi heuristic astar: extreme corridor path" {
    const alloc = testing.allocator;
    const row = [_]u8{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    const grid = [_][]const u8{row[0..]};

    var result = try multiHeuristicAStar(alloc, &grid, .{ .x = 0, .y = 0 }, .{ .x = row.len - 1, .y = 0 });
    defer result.deinit(alloc);

    try testing.expectEqual(row.len - 1, result.cost);
    try testing.expectEqual(row.len, result.path.len);
}
