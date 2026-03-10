//! Bidirectional Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/bidirectional_search.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Performs bidirectional BFS on an unweighted graph to find a shortest path.
/// Returns `null` when no path exists, or when `start/goal` are invalid.
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn bidirectionalSearch(
    allocator: Allocator,
    adj: []const []const usize,
    start: usize,
    goal: usize,
) !?[]usize {
    const n = adj.len;
    if (start == goal) {
        if (start >= n) return null;
        const out = try allocator.alloc(usize, 1);
        out[0] = start;
        return out;
    }
    if (start >= n or goal >= n) return null;

    const none = std.math.maxInt(usize);

    const forward_parent = try allocator.alloc(usize, n);
    defer allocator.free(forward_parent);
    const backward_parent = try allocator.alloc(usize, n);
    defer allocator.free(backward_parent);
    @memset(forward_parent, none);
    @memset(backward_parent, none);
    forward_parent[start] = start;
    backward_parent[goal] = goal;

    var qf = std.ArrayListUnmanaged(usize){};
    defer qf.deinit(allocator);
    var qb = std.ArrayListUnmanaged(usize){};
    defer qb.deinit(allocator);
    var hf: usize = 0;
    var hb: usize = 0;

    try qf.append(allocator, start);
    try qb.append(allocator, goal);

    var intersection: ?usize = null;

    while (hf < qf.items.len and hb < qb.items.len and intersection == null) {
        intersection = try expandSearchLevel(allocator, adj, &qf, &hf, forward_parent, backward_parent);
        if (intersection != null) break;
        intersection = try expandSearchLevel(allocator, adj, &qb, &hb, backward_parent, forward_parent);
    }

    const meet = intersection orelse return null;
    return try constructFullPath(allocator, forward_parent, backward_parent, start, goal, meet);
}

fn expandSearchLevel(
    allocator: Allocator,
    adj: []const []const usize,
    queue: *std.ArrayListUnmanaged(usize),
    head: *usize,
    parents: []usize,
    opposite_parents: []const usize,
) !?usize {
    const level_end = queue.items.len;
    while (head.* < level_end) {
        const current = queue.items[head.*];
        head.* += 1;

        for (adj[current]) |neighbor| {
            if (neighbor >= adj.len) continue;
            if (parents[neighbor] != std.math.maxInt(usize)) continue;

            parents[neighbor] = current;
            try queue.append(allocator, neighbor);

            if (opposite_parents[neighbor] != std.math.maxInt(usize)) {
                return neighbor;
            }
        }
    }
    return null;
}

fn constructPath(
    allocator: Allocator,
    current_start: usize,
    parents: []const usize,
    root: usize,
) ![]usize {
    var out = std.ArrayListUnmanaged(usize){};
    defer out.deinit(allocator);
    var cur = current_start;
    while (true) {
        try out.append(allocator, cur);
        if (cur == root) break;
        const p = parents[cur];
        if (p == std.math.maxInt(usize)) return error.InternalInvariantBroken;
        cur = p;
    }
    return try out.toOwnedSlice(allocator);
}

fn constructFullPath(
    allocator: Allocator,
    forward_parent: []const usize,
    backward_parent: []const usize,
    start: usize,
    goal: usize,
    intersection: usize,
) ![]usize {
    const forward = try constructPath(allocator, intersection, forward_parent, start);
    defer allocator.free(forward);
    std.mem.reverse(usize, forward);

    if (intersection == goal) {
        const out = try allocator.alloc(usize, forward.len);
        @memcpy(out, forward);
        return out;
    }

    const back_start = backward_parent[intersection];
    if (back_start == std.math.maxInt(usize)) return error.InternalInvariantBroken;
    const backward = try constructPath(allocator, back_start, backward_parent, goal);
    defer allocator.free(backward);

    const total = try allocator.alloc(usize, forward.len + backward.len);
    @memcpy(total[0..forward.len], forward);
    @memcpy(total[forward.len..], backward);
    return total;
}

test "bidirectional search: python example path" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{ 0, 3, 4 }, // 1
        &[_]usize{ 0, 5, 6 }, // 2
        &[_]usize{ 1, 7 }, // 3
        &[_]usize{ 1, 8 }, // 4
        &[_]usize{ 2, 9 }, // 5
        &[_]usize{ 2, 10 }, // 6
        &[_]usize{ 3, 11 }, // 7
        &[_]usize{ 4, 11 }, // 8
        &[_]usize{ 5, 11 }, // 9
        &[_]usize{ 6, 11 }, // 10
        &[_]usize{ 7, 8, 9, 10 }, // 11
    };

    const path = (try bidirectionalSearch(alloc, &graph, 0, 11)).?;
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 3, 7, 11 }, path);
}

test "bidirectional search: same node and no path" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{4},
        &[_]usize{3},
    };

    const same = (try bidirectionalSearch(alloc, &graph, 2, 2)).?;
    defer alloc.free(same);
    try testing.expectEqualSlices(usize, &[_]usize{2}, same);

    const none = try bidirectionalSearch(alloc, &graph, 0, 3);
    try testing.expect(none == null);
}

test "bidirectional search: invalid nodes and invalid neighbors" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{0},
    };

    const path = (try bidirectionalSearch(alloc, &graph, 0, 1)).?;
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, path);

    try testing.expect((try bidirectionalSearch(alloc, &graph, 2, 1)) == null);
    try testing.expect((try bidirectionalSearch(alloc, &graph, 0, 5)) == null);
}

test "bidirectional search: extreme chain graph" {
    const alloc = testing.allocator;
    const n: usize = 300;

    const mutable = try alloc.alloc([]usize, n);
    defer {
        for (mutable) |row| alloc.free(row);
        alloc.free(mutable);
    }
    for (0..n) |i| {
        if (i == 0) {
            mutable[i] = try alloc.alloc(usize, 1);
            mutable[i][0] = 1;
        } else if (i + 1 == n) {
            mutable[i] = try alloc.alloc(usize, 1);
            mutable[i][0] = i - 1;
        } else {
            mutable[i] = try alloc.alloc(usize, 2);
            mutable[i][0] = i - 1;
            mutable[i][1] = i + 1;
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable, 0..) |row, i| adj[i] = row;

    const path = (try bidirectionalSearch(alloc, adj, 0, n - 1)).?;
    defer alloc.free(path);
    try testing.expectEqual(n, path.len);
    for (path, 0..) |v, i| try testing.expectEqual(i, v);
}

test "bidirectional search: level-synchronous expansion preserves shortest path" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 3, 4 },
        &[_]usize{ 2, 4, 5 },
        &[_]usize{ 1, 3 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
        &[_]usize{1},
    };

    const path = (try bidirectionalSearch(alloc, &graph, 0, 5)).?;
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 4, 1, 5 }, path);
}
