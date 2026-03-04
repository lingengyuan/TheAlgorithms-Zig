//! Lowest Common Ancestor (Binary Lifting) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/lowest_common_ancestor.py

const std = @import("std");
const testing = std.testing;

pub fn swap(a: usize, b: usize) [2]usize {
    return .{ b, a };
}

/// Builds sparse table where `parent[j][i]` is the 2^j-th ancestor of node `i`.
/// Time complexity: O(n log n), Space complexity: O(1) extra
pub fn createSparse(max_node: usize, parent: [][]usize) void {
    var j: usize = 1;
    while (j < parent.len and ((@as(usize, 1) << @intCast(j)) < max_node)) : (j += 1) {
        var i: usize = 1;
        while (i <= max_node) : (i += 1) {
            parent[j][i] = parent[j - 1][parent[j - 1][i]];
        }
    }
}

/// Returns LCA of `u` and `v`.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn lowestCommonAncestor(
    u_in: usize,
    v_in: usize,
    level: []const i32,
    parent: []const []const usize,
) !usize {
    if (u_in >= level.len or v_in >= level.len) return error.InvalidNode;

    var u = u_in;
    var v = v_in;

    if (level[u] < level[v]) {
        const p = swap(u, v);
        u = p[0];
        v = p[1];
    }

    var i: isize = @as(isize, @intCast(parent.len)) - 1;
    while (i >= 0) : (i -= 1) {
        const step: i32 = @as(i32, 1) << @intCast(i);
        if (level[u] - step >= level[v]) {
            u = parent[@intCast(i)][u];
        }
    }

    if (u == v) return u;

    i = @as(isize, @intCast(parent.len)) - 1;
    while (i >= 0) : (i -= 1) {
        const pu = parent[@intCast(i)][u];
        const pv = parent[@intCast(i)][v];
        if (pu != 0 and pu != pv) {
            u = pu;
            v = pv;
        }
    }

    return parent[0][u];
}

/// BFS from `root` to fill `level` and direct parent row (`parent0`).
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn breadthFirstSearch(
    allocator: std.mem.Allocator,
    level: []i32,
    parent0: []usize,
    max_node: usize,
    graph: []const []const usize,
    root: usize,
) !void {
    if (level.len <= max_node or parent0.len <= max_node or graph.len <= max_node) return error.InvalidInput;
    if (root == 0 or root > max_node) return error.InvalidRoot;

    level[root] = 0;

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    try queue.append(allocator, root);

    var head: usize = 0;
    while (head < queue.items.len) {
        const u = queue.items[head];
        head += 1;

        for (graph[u]) |v| {
            if (v <= max_node and level[v] == -1) {
                level[v] = level[u] + 1;
                parent0[v] = u;
                try queue.append(allocator, v);
            }
        }
    }
}

fn allocUsizeMatrix(allocator: std.mem.Allocator, rows: usize, cols: usize) ![][]usize {
    const matrix = try allocator.alloc([]usize, rows);
    errdefer allocator.free(matrix);

    var i: usize = 0;
    errdefer {
        var j: usize = 0;
        while (j < i) : (j += 1) allocator.free(matrix[j]);
    }

    while (i < rows) : (i += 1) {
        matrix[i] = try allocator.alloc(usize, cols);
        @memset(matrix[i], 0);
    }

    return matrix;
}

fn freeUsizeMatrix(allocator: std.mem.Allocator, matrix: [][]usize) void {
    for (matrix) |row| allocator.free(row);
    allocator.free(matrix);
}

fn naiveLca(u_in: usize, v_in: usize, level: []const i32, parent0: []const usize) usize {
    var u = u_in;
    var v = v_in;

    while (level[u] > level[v]) u = parent0[u];
    while (level[v] > level[u]) v = parent0[v];
    while (u != v) {
        u = parent0[u];
        v = parent0[v];
    }

    return u;
}

test "lowest common ancestor: swap and create sparse" {
    try testing.expectEqual([2]usize{ 3, 2 }, swap(2, 3));
    try testing.expectEqual([2]usize{ 4, 3 }, swap(3, 4));
    try testing.expectEqual([2]usize{ 12, 67 }, swap(67, 12));

    const max_node: usize = 6;
    const parent = try allocUsizeMatrix(testing.allocator, 20, max_node + 1);
    defer freeUsizeMatrix(testing.allocator, parent);

    const base = [_]usize{ 0, 0, 1, 1, 2, 2, 3 };
    @memcpy(parent[0], &base);

    createSparse(max_node, parent);

    try testing.expectEqualSlices(usize, &[_]usize{ 0, 0, 1, 1, 2, 2, 3 }, parent[0]);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 0, 0, 0, 1, 1, 1 }, parent[1]);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 0, 0, 0, 0, 0, 0 }, parent[2]);
}

test "lowest common ancestor: bfs and sample queries" {
    const max_node: usize = 6;

    const level = try testing.allocator.alloc(i32, max_node + 1);
    defer testing.allocator.free(level);
    @memset(level, -1);

    const parent = try allocUsizeMatrix(testing.allocator, 20, max_node + 1);
    defer freeUsizeMatrix(testing.allocator, parent);

    var graph = try testing.allocator.alloc([]const usize, max_node + 1);
    defer testing.allocator.free(graph);
    graph[0] = &[_]usize{};
    graph[1] = &[_]usize{ 2, 3 };
    graph[2] = &[_]usize{ 4, 5 };
    graph[3] = &[_]usize{6};
    graph[4] = &[_]usize{};
    graph[5] = &[_]usize{};
    graph[6] = &[_]usize{};

    try breadthFirstSearch(testing.allocator, level, parent[0], max_node, graph, 1);
    createSparse(max_node, parent);

    try testing.expectEqualSlices(i32, &[_]i32{ -1, 0, 1, 1, 2, 2, 2 }, level);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 0, 1, 1, 2, 2, 3 }, parent[0]);

    try testing.expectEqual(@as(usize, 2), try lowestCommonAncestor(4, 5, level, parent));
    try testing.expectEqual(@as(usize, 1), try lowestCommonAncestor(4, 6, level, parent));
    try testing.expectEqual(@as(usize, 1), try lowestCommonAncestor(2, 3, level, parent));
    try testing.expectEqual(@as(usize, 6), try lowestCommonAncestor(6, 6, level, parent));
}

test "lowest common ancestor: extreme complete tree randomized parity" {
    const max_node: usize = 131_071;

    const level = try testing.allocator.alloc(i32, max_node + 1);
    defer testing.allocator.free(level);
    @memset(level, -1);

    const parent = try allocUsizeMatrix(testing.allocator, 20, max_node + 1);
    defer freeUsizeMatrix(testing.allocator, parent);

    level[0] = -1;
    parent[0][0] = 0;

    var i: usize = 1;
    while (i <= max_node) : (i += 1) {
        parent[0][i] = i / 2;

        var x = i;
        var d: i32 = 0;
        while (x > 1) : (d += 1) {
            x /= 2;
        }
        level[i] = d;
    }

    createSparse(max_node, parent);

    var prng = std.Random.DefaultPrng.init(0x9E3779B97F4A7C15);
    const random = prng.random();

    var q: usize = 0;
    while (q < 20_000) : (q += 1) {
        const u = random.uintLessThan(usize, max_node) + 1;
        const v = random.uintLessThan(usize, max_node) + 1;

        const got = try lowestCommonAncestor(u, v, level, parent);
        const want = naiveLca(u, v, level, parent[0]);
        try testing.expectEqual(want, got);
    }
}
