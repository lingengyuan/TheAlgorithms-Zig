//! Disjoint Set (Union-Find) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/tree/master/data_structures/disjoint_set

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Disjoint-set data structure with path compression and union by rank.
/// `find` / `union` amortized time complexity: O(alpha(n)), space: O(n)
pub const DisjointSet = struct {
    const Self = @This();

    allocator: Allocator,
    parent: []usize,
    rank: []u8,
    components: usize,

    pub fn init(allocator: Allocator, n: usize) !Self {
        const parent = try allocator.alloc(usize, n);
        errdefer allocator.free(parent);
        const rank = try allocator.alloc(u8, n);

        for (0..n) |i| {
            parent[i] = i;
            rank[i] = 0;
        }

        return .{
            .allocator = allocator,
            .parent = parent,
            .rank = rank,
            .components = n,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.parent);
        self.allocator.free(self.rank);
        self.* = undefined;
    }

    pub fn size(self: *const Self) usize {
        return self.parent.len;
    }

    pub fn componentCount(self: *const Self) usize {
        return self.components;
    }

    pub fn find(self: *Self, x: usize) !usize {
        if (x >= self.parent.len) return error.IndexOutOfBounds;
        return self.findNoCheck(x);
    }

    pub fn connected(self: *Self, a: usize, b: usize) !bool {
        if (a >= self.parent.len or b >= self.parent.len) return error.IndexOutOfBounds;
        return self.findNoCheck(a) == self.findNoCheck(b);
    }

    /// Returns true if two different components were merged; false when already connected.
    pub fn unionSets(self: *Self, a: usize, b: usize) !bool {
        if (a >= self.parent.len or b >= self.parent.len) return error.IndexOutOfBounds;

        var root_a = self.findNoCheck(a);
        var root_b = self.findNoCheck(b);
        if (root_a == root_b) return false;

        if (self.rank[root_a] < self.rank[root_b]) {
            std.mem.swap(usize, &root_a, &root_b);
        }

        self.parent[root_b] = root_a;
        if (self.rank[root_a] == self.rank[root_b]) {
            self.rank[root_a] +%= 1;
        }
        self.components -= 1;
        return true;
    }

    fn findNoCheck(self: *Self, x: usize) usize {
        var root = x;
        while (self.parent[root] != root) {
            root = self.parent[root];
        }

        var cur = x;
        while (self.parent[cur] != cur) {
            const next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        return root;
    }
};

test "disjoint set: basic unions and connectivity" {
    var ds = try DisjointSet.init(testing.allocator, 8);
    defer ds.deinit();

    try testing.expectEqual(@as(usize, 8), ds.componentCount());
    try testing.expect(try ds.unionSets(0, 1));
    try testing.expect(try ds.unionSets(1, 2));
    try testing.expect(try ds.unionSets(4, 5));
    try testing.expect(try ds.connected(0, 2));
    try testing.expect(!try ds.connected(0, 4));
    try testing.expectEqual(@as(usize, 5), ds.componentCount());
}

test "disjoint set: union already connected" {
    var ds = try DisjointSet.init(testing.allocator, 4);
    defer ds.deinit();

    try testing.expect(try ds.unionSets(0, 1));
    try testing.expect(!try ds.unionSets(1, 0));
    try testing.expectEqual(@as(usize, 3), ds.componentCount());
}

test "disjoint set: path compression keeps same root" {
    var ds = try DisjointSet.init(testing.allocator, 6);
    defer ds.deinit();

    _ = try ds.unionSets(0, 1);
    _ = try ds.unionSets(1, 2);
    _ = try ds.unionSets(2, 3);

    const root_a = try ds.find(0);
    const root_b = try ds.find(3);
    try testing.expectEqual(root_a, root_b);
    try testing.expectEqual(root_a, try ds.find(1));
}

test "disjoint set: index bounds" {
    var ds = try DisjointSet.init(testing.allocator, 3);
    defer ds.deinit();

    try testing.expectError(error.IndexOutOfBounds, ds.find(3));
    try testing.expectError(error.IndexOutOfBounds, ds.connected(0, 3));
    try testing.expectError(error.IndexOutOfBounds, ds.unionSets(3, 1));
}
