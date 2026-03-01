//! Eulerian Path/Circuit (Undirected) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/eulerian_path_and_circuit_for_undirected_graph.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const EulerKind = enum {
    circuit,
    path,
};

pub const EulerResult = struct {
    kind: EulerKind,
    path: []usize,
};

const PairCount = struct {
    u: usize,
    v: usize,
    count: usize,
};

const UndirectedEdge = struct {
    u: usize,
    v: usize,
};

/// Finds an Eulerian path or circuit in an undirected graph using Hierholzer's algorithm.
/// Input is adjacency-list based; invalid neighbor indices are ignored.
/// Returns `error.NotEulerian` when no Eulerian path/circuit exists.
/// Time complexity: O(V + E), Space complexity: O(V + E)
pub fn findEulerianPathOrCircuit(allocator: Allocator, adj: []const []const usize) !EulerResult {
    const n = adj.len;
    if (n == 0) {
        return .{
            .kind = .circuit,
            .path = try allocator.alloc(usize, 0),
        };
    }

    var directed_counts = std.AutoHashMap(u128, usize).init(allocator);
    defer directed_counts.deinit();

    for (adj, 0..) |neighbors, u| {
        for (neighbors) |v| {
            if (v >= n) continue;
            const key = encodePair(u, v);
            const gop = try directed_counts.getOrPut(key);
            if (!gop.found_existing) {
                gop.value_ptr.* = 0;
            }
            gop.value_ptr.* += 1;
        }
    }

    var pairs = std.ArrayListUnmanaged(PairCount){};
    defer pairs.deinit(allocator);

    var it = directed_counts.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const u = decodeA(key);
        const v = decodeB(key);
        if (u > v) continue;

        var undirected_count: usize = 0;
        if (u == v) {
            undirected_count = entry.value_ptr.* / 2;
        } else {
            const reverse_count = directed_counts.get(encodePair(v, u)) orelse 0;
            undirected_count = @min(entry.value_ptr.*, reverse_count);
        }

        if (undirected_count > 0) {
            try pairs.append(allocator, .{
                .u = u,
                .v = v,
                .count = undirected_count,
            });
        }
    }

    std.sort.heap(PairCount, pairs.items, {}, lessPair);

    var edge_total: usize = 0;
    for (pairs.items) |pair| {
        edge_total += pair.count;
    }

    if (edge_total == 0) {
        return .{
            .kind = .circuit,
            .path = try allocator.alloc(usize, 0),
        };
    }

    const edges = try allocator.alloc(UndirectedEdge, edge_total);
    defer allocator.free(edges);

    const incidence = try allocator.alloc(std.ArrayListUnmanaged(usize), n);
    defer {
        for (0..n) |i| incidence[i].deinit(allocator);
        allocator.free(incidence);
    }
    for (0..n) |i| incidence[i] = .{};

    const degree = try allocator.alloc(usize, n);
    defer allocator.free(degree);
    @memset(degree, 0);

    var edge_idx: usize = 0;
    for (pairs.items) |pair| {
        for (0..pair.count) |_| {
            edges[edge_idx] = .{ .u = pair.u, .v = pair.v };
            try incidence[pair.u].append(allocator, edge_idx);
            if (pair.v != pair.u) {
                try incidence[pair.v].append(allocator, edge_idx);
                degree[pair.u] += 1;
                degree[pair.v] += 1;
            } else {
                degree[pair.u] += 2;
            }
            edge_idx += 1;
        }
    }

    var odd_count: usize = 0;
    var start: usize = std.math.maxInt(usize);
    for (0..n) |node| {
        if (degree[node] % 2 == 1) {
            odd_count += 1;
            if (start == std.math.maxInt(usize)) start = node;
        }
    }

    if (odd_count != 0 and odd_count != 2) return error.NotEulerian;

    if (start == std.math.maxInt(usize)) {
        for (0..n) |node| {
            if (degree[node] > 0) {
                start = node;
                break;
            }
        }
    }
    if (start == std.math.maxInt(usize)) {
        return .{
            .kind = .circuit,
            .path = try allocator.alloc(usize, 0),
        };
    }

    try ensureConnected(allocator, edges, incidence, degree, start);

    const edge_used = try allocator.alloc(bool, edge_total);
    defer allocator.free(edge_used);
    @memset(edge_used, false);

    const cursor = try allocator.alloc(usize, n);
    defer allocator.free(cursor);
    @memset(cursor, 0);

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);
    var reversed = std.ArrayListUnmanaged(usize){};
    defer reversed.deinit(allocator);

    try stack.append(allocator, start);

    while (stack.items.len > 0) {
        const cur = stack.items[stack.items.len - 1];
        var advanced = false;

        while (cursor[cur] < incidence[cur].items.len) {
            const eid = incidence[cur].items[cursor[cur]];
            cursor[cur] += 1;
            if (edge_used[eid]) continue;

            edge_used[eid] = true;
            const edge = edges[eid];
            const next = if (edge.u == cur) edge.v else edge.u;
            try stack.append(allocator, next);
            advanced = true;
            break;
        }

        if (!advanced) {
            const node = stack.pop().?;
            try reversed.append(allocator, node);
        }
    }

    if (reversed.items.len != edge_total + 1) return error.NotEulerian;
    std.mem.reverse(usize, reversed.items);

    return .{
        .kind = if (odd_count == 0) .circuit else .path,
        .path = try reversed.toOwnedSlice(allocator),
    };
}

fn lessPair(_: void, a: PairCount, b: PairCount) bool {
    if (a.u != b.u) return a.u < b.u;
    if (a.v != b.v) return a.v < b.v;
    return a.count < b.count;
}

fn encodePair(a: usize, b: usize) u128 {
    return (@as(u128, @intCast(a)) << 64) | @as(u128, @intCast(b));
}

fn decodeA(key: u128) usize {
    return @as(usize, @intCast(key >> 64));
}

fn decodeB(key: u128) usize {
    return @as(usize, @intCast(key & 0xFFFF_FFFF_FFFF_FFFF));
}

fn ensureConnected(
    allocator: Allocator,
    edges: []const UndirectedEdge,
    incidence: []const std.ArrayListUnmanaged(usize),
    degree: []const usize,
    start: usize,
) !void {
    const visited = try allocator.alloc(bool, degree.len);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);
    try stack.append(allocator, start);
    visited[start] = true;

    while (stack.items.len > 0) {
        const cur = stack.pop().?;
        for (incidence[cur].items) |eid| {
            const edge = edges[eid];
            const nb = if (edge.u == cur) edge.v else edge.u;
            if (!visited[nb]) {
                visited[nb] = true;
                try stack.append(allocator, nb);
            }
        }
    }

    for (degree, 0..) |deg, node| {
        if (deg > 0 and !visited[node]) return error.NotEulerian;
    }
}

test "euler undirected: simple circuit" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
    };

    const result = try findEulerianPathOrCircuit(alloc, &adj);
    defer alloc.free(result.path);

    try testing.expectEqual(EulerKind.circuit, result.kind);
    try testing.expectEqual(@as(usize, 4), result.path.len);
    try testing.expectEqual(@as(usize, 0), result.path[0]);
    try testing.expectEqual(result.path[0], result.path[result.path.len - 1]);
}

test "euler undirected: simple path with two odd nodes" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{1},
        &[_]usize{ 0, 2 },
        &[_]usize{ 1, 3 },
        &[_]usize{2},
    };

    const result = try findEulerianPathOrCircuit(alloc, &adj);
    defer alloc.free(result.path);

    try testing.expectEqual(EulerKind.path, result.kind);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, result.path);
}

test "euler undirected: odd degree count > 2 is not eulerian" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2, 3 },
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{0},
    };

    try testing.expectError(error.NotEulerian, findEulerianPathOrCircuit(alloc, &adj));
}

test "euler undirected: disconnected non-zero degree graph is not eulerian" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
        &[_]usize{ 4, 5 },
        &[_]usize{ 3, 5 },
        &[_]usize{ 3, 4 },
    };

    try testing.expectError(error.NotEulerian, findEulerianPathOrCircuit(alloc, &adj));
}

test "euler undirected: invalid neighbor index is ignored" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 9 },
        &[_]usize{0},
        &[_]usize{},
    };

    const result = try findEulerianPathOrCircuit(alloc, &adj);
    defer alloc.free(result.path);

    try testing.expectEqual(EulerKind.path, result.kind);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, result.path);
}

test "euler undirected: parallel edges form circuit" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{
        &[_]usize{ 1, 1 },
        &[_]usize{ 0, 0 },
    };

    const result = try findEulerianPathOrCircuit(alloc, &adj);
    defer alloc.free(result.path);

    try testing.expectEqual(EulerKind.circuit, result.kind);
    try testing.expectEqual(@as(usize, 3), result.path.len);
    try testing.expectEqual(result.path[0], result.path[result.path.len - 1]);
}

test "euler undirected: empty graph returns empty circuit" {
    const alloc = testing.allocator;
    const adj = [_][]const usize{};

    const result = try findEulerianPathOrCircuit(alloc, &adj);
    defer alloc.free(result.path);

    try testing.expectEqual(EulerKind.circuit, result.kind);
    try testing.expectEqual(@as(usize, 0), result.path.len);
}
