//! Bidirectional Breadth-First Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/bidirectional_breadth_first_search.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Finds a path from `start` to `goal` in an unweighted graph via bidirectional BFS.
/// Returns a path slice including endpoints.
/// If no path is found, returns `[start]` to match the Python reference behavior.
/// Invalid neighbor indices are ignored.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn bidirectionalBfsPath(
    allocator: Allocator,
    adj: []const []const usize,
    start: usize,
    goal: usize,
) ![]usize {
    const n = adj.len;
    if (start >= n or goal >= n) return error.InvalidNode;

    if (start == goal) {
        const out = try allocator.alloc(usize, 1);
        out[0] = start;
        return out;
    }

    const none = std.math.maxInt(usize);
    const visited_start = try allocator.alloc(bool, n);
    defer allocator.free(visited_start);
    const visited_goal = try allocator.alloc(bool, n);
    defer allocator.free(visited_goal);
    const parent_start = try allocator.alloc(usize, n);
    defer allocator.free(parent_start);
    const parent_goal = try allocator.alloc(usize, n);
    defer allocator.free(parent_goal);

    @memset(visited_start, false);
    @memset(visited_goal, false);
    @memset(parent_start, none);
    @memset(parent_goal, none);

    var q_start = std.ArrayListUnmanaged(usize){};
    defer q_start.deinit(allocator);
    var q_goal = std.ArrayListUnmanaged(usize){};
    defer q_goal.deinit(allocator);

    var head_start: usize = 0;
    var head_goal: usize = 0;

    visited_start[start] = true;
    visited_goal[goal] = true;
    parent_start[start] = start;
    parent_goal[goal] = goal;
    try q_start.append(allocator, start);
    try q_goal.append(allocator, goal);

    var meeting: ?usize = null;

    while (head_start < q_start.items.len and head_goal < q_goal.items.len and meeting == null) {
        const cur_start = q_start.items[head_start];
        head_start += 1;
        for (adj[cur_start]) |next| {
            if (next >= n) continue;
            if (!visited_start[next]) {
                visited_start[next] = true;
                parent_start[next] = cur_start;
                try q_start.append(allocator, next);
            }
            if (visited_goal[next]) {
                meeting = next;
                break;
            }
        }
        if (meeting != null) break;

        const cur_goal = q_goal.items[head_goal];
        head_goal += 1;
        for (adj[cur_goal]) |next| {
            if (next >= n) continue;
            if (!visited_goal[next]) {
                visited_goal[next] = true;
                parent_goal[next] = cur_goal;
                try q_goal.append(allocator, next);
            }
            if (visited_start[next]) {
                meeting = next;
                break;
            }
        }
    }

    if (meeting == null) {
        const out = try allocator.alloc(usize, 1);
        out[0] = start;
        return out;
    }

    const meet = meeting.?;
    var left_reversed = std.ArrayListUnmanaged(usize){};
    defer left_reversed.deinit(allocator);

    var cur = meet;
    while (cur != start) {
        try left_reversed.append(allocator, cur);
        const p = parent_start[cur];
        if (p == none) return error.InternalInvariantBroken;
        cur = p;
    }
    try left_reversed.append(allocator, start);
    std.mem.reverse(usize, left_reversed.items);

    var right = std.ArrayListUnmanaged(usize){};
    defer right.deinit(allocator);
    cur = meet;
    while (cur != goal) {
        const p = parent_goal[cur];
        if (p == none) return error.InternalInvariantBroken;
        cur = p;
        try right.append(allocator, cur);
    }

    const output_len = left_reversed.items.len + right.items.len;
    const out = try allocator.alloc(usize, output_len);
    @memcpy(out[0..left_reversed.items.len], left_reversed.items);
    @memcpy(out[left_reversed.items.len..], right.items);
    return out;
}

test "bidirectional bfs: path found in connected graph" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 }, // 0
        &[_]usize{ 0, 3 }, // 1
        &[_]usize{ 0, 4 }, // 2
        &[_]usize{ 1, 5 }, // 3
        &[_]usize{ 2, 5 }, // 4
        &[_]usize{ 3, 4 }, // 5
    };

    const path = try bidirectionalBfsPath(alloc, &adj, 0, 5);
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 3, 5 }, path);
}

test "bidirectional bfs: no path returns start only" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{},
    };

    const path = try bidirectionalBfsPath(alloc, &adj, 0, 2);
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{0}, path);
}

test "bidirectional bfs: same node and invalid node handling" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
    };

    const same = try bidirectionalBfsPath(alloc, &adj, 1, 1);
    defer alloc.free(same);
    try testing.expectEqualSlices(usize, &[_]usize{1}, same);

    try testing.expectError(error.InvalidNode, bidirectionalBfsPath(alloc, &adj, 5, 0));
}

test "bidirectional bfs: invalid neighbors ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{0},
    };

    const path = try bidirectionalBfsPath(alloc, &adj, 0, 1);
    defer alloc.free(path);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, path);
}

test "bidirectional bfs: extreme long chain" {
    const alloc = testing.allocator;
    const n: usize = 257;

    const mutable_adj = try alloc.alloc([]usize, n);
    defer {
        for (mutable_adj) |row| alloc.free(row);
        alloc.free(mutable_adj);
    }

    for (0..n) |i| {
        if (i == 0) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = 1;
        } else if (i + 1 == n) {
            mutable_adj[i] = try alloc.alloc(usize, 1);
            mutable_adj[i][0] = i - 1;
        } else {
            mutable_adj[i] = try alloc.alloc(usize, 2);
            mutable_adj[i][0] = i - 1;
            mutable_adj[i][1] = i + 1;
        }
    }

    const adj = try alloc.alloc([]const usize, n);
    defer alloc.free(adj);
    for (mutable_adj, 0..) |row, i| adj[i] = row;

    const path = try bidirectionalBfsPath(alloc, adj, 0, n - 1);
    defer alloc.free(path);

    try testing.expectEqual(n, path.len);
    for (path, 0..) |v, i| try testing.expectEqual(i, v);
}
