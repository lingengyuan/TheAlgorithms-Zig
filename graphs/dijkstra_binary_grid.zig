//! Dijkstra on Binary Grid - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra_binary_grid.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Point = struct {
    row: usize,
    col: usize,
};

pub const GridPathResult = struct {
    distance: ?usize,
    path: []Point,

    pub fn deinit(self: GridPathResult, allocator: Allocator) void {
        allocator.free(self.path);
    }
};

/// Computes shortest path in a binary grid where walkable cells are `1`.
/// Movement cost is 1 per step; optional diagonal moves are supported.
/// Returns `{ .distance = null, .path = [] }` when destination is unreachable.
/// Source cell is accepted even if blocked, matching the Python reference behavior.
/// Time complexity: O((R*C)^2), Space complexity: O(R*C)
pub fn dijkstraBinaryGrid(
    allocator: Allocator,
    grid: []const []const u8,
    source: Point,
    destination: Point,
    allow_diagonal: bool,
) !GridPathResult {
    const rows = grid.len;
    if (rows == 0) return error.InvalidGrid;
    const cols = grid[0].len;
    if (cols == 0) return error.InvalidGrid;

    for (grid) |row| {
        if (row.len != cols) return error.InvalidGrid;
    }

    if (source.row >= rows or source.col >= cols) return error.InvalidPoint;
    if (destination.row >= rows or destination.col >= cols) return error.InvalidPoint;

    const total_cells_mul = @mulWithOverflow(rows, cols);
    if (total_cells_mul[1] != 0) return error.Overflow;
    const total_cells = total_cells_mul[0];

    const dist = try allocator.alloc(usize, total_cells);
    defer allocator.free(dist);
    const prev = try allocator.alloc(usize, total_cells);
    defer allocator.free(prev);
    const visited = try allocator.alloc(bool, total_cells);
    defer allocator.free(visited);

    const none: usize = std.math.maxInt(usize);
    @memset(dist, none);
    @memset(prev, none);
    @memset(visited, false);

    const source_idx = source.row * cols + source.col;
    const dest_idx = destination.row * cols + destination.col;
    dist[source_idx] = 0;
    prev[source_idx] = source_idx;

    if (source_idx == dest_idx) {
        const path = try allocator.alloc(Point, 1);
        path[0] = source;
        return .{ .distance = 0, .path = path };
    }

    while (true) {
        var current_idx: ?usize = null;
        var current_best: usize = none;

        for (0..total_cells) |idx| {
            if (visited[idx]) continue;
            if (dist[idx] < current_best) {
                current_best = dist[idx];
                current_idx = idx;
            }
        }

        if (current_idx == null or current_best == none) break;
        const u = current_idx.?;
        if (u == dest_idx) break;
        visited[u] = true;

        const r = u / cols;
        const c = u % cols;

        try relaxNeighbor(allocator, grid, allow_diagonal, r, c, rows, cols, dist, prev, u);
    }

    if (dist[dest_idx] == none) {
        return .{
            .distance = null,
            .path = try allocator.alloc(Point, 0),
        };
    }

    var reversed = std.ArrayListUnmanaged(Point){};
    defer reversed.deinit(allocator);
    var cur = dest_idx;

    while (cur != source_idx) {
        try reversed.append(allocator, .{
            .row = cur / cols,
            .col = cur % cols,
        });
        const p = prev[cur];
        if (p == none) return error.InternalInvariantBroken;
        cur = p;
    }
    try reversed.append(allocator, source);
    std.mem.reverse(Point, reversed.items);

    return .{
        .distance = dist[dest_idx],
        .path = try reversed.toOwnedSlice(allocator),
    };
}

fn relaxOne(
    grid: []const []const u8,
    rows: usize,
    cols: usize,
    from_idx: usize,
    to_row: isize,
    to_col: isize,
    dist: []usize,
    prev: []usize,
) !void {
    if (to_row < 0 or to_col < 0) return;
    const r = @as(usize, @intCast(to_row));
    const c = @as(usize, @intCast(to_col));
    if (r >= rows or c >= cols) return;
    if (grid[r][c] != 1) return;

    const to_idx = r * cols + c;
    const sum = @addWithOverflow(dist[from_idx], 1);
    if (sum[1] != 0) return error.Overflow;
    const candidate = sum[0];

    if (candidate < dist[to_idx]) {
        dist[to_idx] = candidate;
        prev[to_idx] = from_idx;
    }
}

fn relaxNeighbor(
    allocator: Allocator,
    grid: []const []const u8,
    allow_diagonal: bool,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    dist: []usize,
    prev: []usize,
    from_idx: usize,
) !void {
    _ = allocator;
    const r = @as(isize, @intCast(row));
    const c = @as(isize, @intCast(col));

    try relaxOne(grid, rows, cols, from_idx, r - 1, c, dist, prev);
    try relaxOne(grid, rows, cols, from_idx, r + 1, c, dist, prev);
    try relaxOne(grid, rows, cols, from_idx, r, c - 1, dist, prev);
    try relaxOne(grid, rows, cols, from_idx, r, c + 1, dist, prev);

    if (allow_diagonal) {
        try relaxOne(grid, rows, cols, from_idx, r - 1, c - 1, dist, prev);
        try relaxOne(grid, rows, cols, from_idx, r - 1, c + 1, dist, prev);
        try relaxOne(grid, rows, cols, from_idx, r + 1, c - 1, dist, prev);
        try relaxOne(grid, rows, cols, from_idx, r + 1, c + 1, dist, prev);
    }
}

test "dijkstra binary grid: python examples" {
    const alloc = testing.allocator;

    const grid1 = [_][]const u8{
        &[_]u8{ 1, 1, 1 },
        &[_]u8{ 0, 1, 0 },
        &[_]u8{ 0, 1, 1 },
    };
    var r1 = try dijkstraBinaryGrid(alloc, &grid1, .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 2 }, false);
    defer r1.deinit(alloc);
    try testing.expectEqual(@as(?usize, 4), r1.distance);
    try testing.expectEqual(@as(usize, 5), r1.path.len);
    try testing.expectEqual(@as(usize, 2), r1.path[r1.path.len - 1].row);
    try testing.expectEqual(@as(usize, 2), r1.path[r1.path.len - 1].col);

    var r2 = try dijkstraBinaryGrid(alloc, &grid1, .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 2 }, true);
    defer r2.deinit(alloc);
    try testing.expectEqual(@as(?usize, 2), r2.distance);
    try testing.expectEqual(@as(usize, 3), r2.path.len);

    const grid3 = [_][]const u8{
        &[_]u8{ 1, 1, 1 },
        &[_]u8{ 0, 0, 1 },
        &[_]u8{ 0, 1, 1 },
    };
    var r3 = try dijkstraBinaryGrid(alloc, &grid3, .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 2 }, false);
    defer r3.deinit(alloc);
    try testing.expectEqual(@as(?usize, 4), r3.distance);
    try testing.expectEqual(@as(usize, 5), r3.path.len);
}

test "dijkstra binary grid: unreachable and source equals destination" {
    const alloc = testing.allocator;
    const unreachable_grid = [_][]const u8{
        &[_]u8{ 1, 0 },
        &[_]u8{ 0, 1 },
    };
    var r1 = try dijkstraBinaryGrid(alloc, &unreachable_grid, .{ .row = 0, .col = 0 }, .{ .row = 1, .col = 1 }, false);
    defer r1.deinit(alloc);
    try testing.expectEqual(@as(?usize, null), r1.distance);
    try testing.expectEqual(@as(usize, 0), r1.path.len);

    const blocked_single = [_][]const u8{
        &[_]u8{0},
    };
    var r2 = try dijkstraBinaryGrid(alloc, &blocked_single, .{ .row = 0, .col = 0 }, .{ .row = 0, .col = 0 }, false);
    defer r2.deinit(alloc);
    try testing.expectEqual(@as(?usize, 0), r2.distance);
    try testing.expectEqual(@as(usize, 1), r2.path.len);
}

test "dijkstra binary grid: invalid grid and invalid point" {
    const alloc = testing.allocator;
    const ragged = [_][]const u8{
        &[_]u8{ 1, 1 },
        &[_]u8{1},
    };
    try testing.expectError(
        error.InvalidGrid,
        dijkstraBinaryGrid(alloc, &ragged, .{ .row = 0, .col = 0 }, .{ .row = 1, .col = 0 }, false),
    );

    const valid = [_][]const u8{
        &[_]u8{ 1, 1 },
        &[_]u8{ 1, 1 },
    };
    try testing.expectError(
        error.InvalidPoint,
        dijkstraBinaryGrid(alloc, &valid, .{ .row = 2, .col = 0 }, .{ .row = 1, .col = 1 }, false),
    );
}

test "dijkstra binary grid: extreme large open grid" {
    const alloc = testing.allocator;
    const rows: usize = 64;
    const cols: usize = 64;

    const mutable_grid = try alloc.alloc([]u8, rows);
    defer {
        for (mutable_grid) |row| alloc.free(row);
        alloc.free(mutable_grid);
    }

    for (0..rows) |r| {
        mutable_grid[r] = try alloc.alloc(u8, cols);
        @memset(mutable_grid[r], 1);
    }

    const grid = try alloc.alloc([]const u8, rows);
    defer alloc.free(grid);
    for (mutable_grid, 0..) |row, i| grid[i] = row;

    var result = try dijkstraBinaryGrid(alloc, grid, .{ .row = 0, .col = 0 }, .{ .row = rows - 1, .col = cols - 1 }, false);
    defer result.deinit(alloc);

    try testing.expectEqual(@as(?usize, (rows - 1) + (cols - 1)), result.distance);
    try testing.expectEqual(@as(usize, (rows - 1) + (cols - 1) + 1), result.path.len);
}
