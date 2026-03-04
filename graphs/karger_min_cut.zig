//! Karger Minimum Cut - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/karger.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const UndirectedEdge = struct {
    u: usize,
    v: usize,
};

/// Computes a minimum cut approximation via Karger's randomized contraction algorithm.
/// Input graph is adjacency-list based and treated as undirected.
/// Invalid neighbors and self loops are ignored.
/// `trials` controls repetition; more trials increases confidence.
/// Time complexity: O(trials * V * E), Space complexity: O(V + E)
pub fn kargerMinCut(
    allocator: Allocator,
    graph: []const []const usize,
    random: std.Random,
    trials: usize,
) !usize {
    if (trials == 0) return error.InvalidTrials;

    const n = graph.len;
    if (n < 2) return 0;

    var edges = std.ArrayListUnmanaged(UndirectedEdge){};
    defer edges.deinit(allocator);
    try collectUniqueUndirectedEdges(allocator, graph, &edges);

    if (edges.items.len == 0) return 0;

    var best: usize = std.math.maxInt(usize);

    for (0..trials) |_| {
        var uf = try UnionFind.init(allocator, n);
        defer uf.deinit(allocator);

        var components = n;
        while (components > 2) {
            var crossing_count: usize = 0;
            for (edges.items) |e| {
                if (uf.find(e.u) != uf.find(e.v)) crossing_count += 1;
            }
            if (crossing_count == 0) break;

            const pick_rank = random.uintLessThan(usize, crossing_count);
            var seen: usize = 0;
            var pick = edges.items[0];

            for (edges.items) |e| {
                if (uf.find(e.u) == uf.find(e.v)) continue;
                if (seen == pick_rank) {
                    pick = e;
                    break;
                }
                seen += 1;
            }

            if (uf.unionSets(pick.u, pick.v)) components -= 1;
        }

        var cut: usize = 0;
        for (edges.items) |e| {
            if (uf.find(e.u) != uf.find(e.v)) cut += 1;
        }

        if (cut < best) best = cut;
    }

    return if (best == std.math.maxInt(usize)) 0 else best;
}

fn collectUniqueUndirectedEdges(
    allocator: Allocator,
    graph: []const []const usize,
    out: *std.ArrayListUnmanaged(UndirectedEdge),
) !void {
    const n = graph.len;
    const seen = try allocator.alloc(bool, n * n);
    defer allocator.free(seen);
    @memset(seen, false);

    for (0..n) |u| {
        for (graph[u]) |v| {
            if (v >= n or v == u) continue;
            const a = @min(u, v);
            const b = @max(u, v);
            const index = a * n + b;
            if (seen[index]) continue;
            seen[index] = true;
            try out.append(allocator, .{ .u = a, .v = b });
        }
    }
}

const UnionFind = struct {
    parent: []usize,
    rank: []u8,

    fn init(allocator: Allocator, n: usize) !UnionFind {
        const parent = try allocator.alloc(usize, n);
        errdefer allocator.free(parent);
        const rank = try allocator.alloc(u8, n);
        errdefer allocator.free(rank);

        for (0..n) |i| {
            parent[i] = i;
            rank[i] = 0;
        }

        return .{ .parent = parent, .rank = rank };
    }

    fn deinit(self: *UnionFind, allocator: Allocator) void {
        allocator.free(self.parent);
        allocator.free(self.rank);
    }

    fn find(self: *UnionFind, x: usize) usize {
        if (self.parent[x] != x) self.parent[x] = self.find(self.parent[x]);
        return self.parent[x];
    }

    fn unionSets(self: *UnionFind, a: usize, b: usize) bool {
        var ra = self.find(a);
        var rb = self.find(b);
        if (ra == rb) return false;

        if (self.rank[ra] < self.rank[rb]) std.mem.swap(usize, &ra, &rb);
        self.parent[rb] = ra;
        if (self.rank[ra] == self.rank[rb]) self.rank[ra] += 1;
        return true;
    }
};

test "karger min cut: two node graph" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);

    const graph = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
    };

    const cut = try kargerMinCut(alloc, &graph, prng.random(), 5);
    try testing.expectEqual(@as(usize, 1), cut);
}

test "karger min cut: disconnected graph is zero" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(2);

    const graph = [_][]const usize{
        &[_]usize{1},
        &[_]usize{0},
        &[_]usize{},
    };

    const cut = try kargerMinCut(alloc, &graph, prng.random(), 8);
    try testing.expectEqual(@as(usize, 0), cut);
}

test "karger min cut: triangle graph" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);

    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
    };

    const cut = try kargerMinCut(alloc, &graph, prng.random(), 20);
    try testing.expectEqual(@as(usize, 2), cut);
}

test "karger min cut: ignores invalid neighbors" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);

    const graph = [_][]const usize{
        &[_]usize{ 1, 99 },
        &[_]usize{0},
    };

    const cut = try kargerMinCut(alloc, &graph, prng.random(), 5);
    try testing.expectEqual(@as(usize, 1), cut);
}

test "karger min cut: extreme chain" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);

    const n: usize = 80;
    const graph = try alloc.alloc([]const usize, n);
    defer alloc.free(graph);

    const forward = try alloc.alloc(usize, n - 1);
    defer alloc.free(forward);
    for (0..n - 1) |i| forward[i] = i + 1;

    for (0..n - 1) |i| graph[i] = forward[i .. i + 1];
    graph[n - 1] = &[_]usize{};

    const cut = try kargerMinCut(alloc, graph, prng.random(), 20);
    try testing.expectEqual(@as(usize, 1), cut);
}
