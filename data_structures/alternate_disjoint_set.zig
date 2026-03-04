//! Alternate Disjoint Set (Union by Rank + Path Compression) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/disjoint_set/alternate_disjoint_set.py

const std = @import("std");
const testing = std.testing;

pub const DisjointSet = struct {
    set_counts: []usize,
    max_set: usize,
    ranks: []usize,
    parents: []usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, counts: []const usize) !DisjointSet {
        if (counts.len == 0) return error.EmptyInput;

        const set_counts = try allocator.alloc(usize, counts.len);
        errdefer allocator.free(set_counts);
        @memcpy(set_counts, counts);

        const ranks = try allocator.alloc(usize, counts.len);
        errdefer allocator.free(ranks);
        @memset(ranks, 1);

        const parents = try allocator.alloc(usize, counts.len);
        errdefer allocator.free(parents);
        for (parents, 0..) |*p, i| p.* = i;

        var max_set = set_counts[0];
        for (set_counts[1..]) |v| max_set = @max(max_set, v);

        return .{
            .set_counts = set_counts,
            .max_set = max_set,
            .ranks = ranks,
            .parents = parents,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DisjointSet) void {
        self.allocator.free(self.set_counts);
        self.allocator.free(self.ranks);
        self.allocator.free(self.parents);
    }

    /// Finds representative parent with path compression.
    /// Time complexity: amortized O(alpha(n)), Space complexity: O(alpha(n)) recursion
    pub fn getParent(self: *DisjointSet, disj_set: usize) !usize {
        if (disj_set >= self.parents.len) return error.IndexOutOfBounds;

        if (self.parents[disj_set] == disj_set) return disj_set;
        self.parents[disj_set] = try self.getParent(self.parents[disj_set]);
        return self.parents[disj_set];
    }

    /// Merges two sets using union by rank.
    /// Returns true if merged, false if already in same set.
    /// Time complexity: amortized O(alpha(n)), Space complexity: O(alpha(n))
    pub fn merge(self: *DisjointSet, src: usize, dst: usize) !bool {
        const src_parent = try self.getParent(src);
        const dst_parent = try self.getParent(dst);

        if (src_parent == dst_parent) return false;

        if (self.ranks[dst_parent] >= self.ranks[src_parent]) {
            self.set_counts[dst_parent] += self.set_counts[src_parent];
            self.set_counts[src_parent] = 0;
            self.parents[src_parent] = dst_parent;
            if (self.ranks[dst_parent] == self.ranks[src_parent]) {
                self.ranks[dst_parent] += 1;
            }
            self.max_set = @max(self.max_set, self.set_counts[dst_parent]);
        } else {
            self.set_counts[src_parent] += self.set_counts[dst_parent];
            self.set_counts[dst_parent] = 0;
            self.parents[dst_parent] = src_parent;
            self.max_set = @max(self.max_set, self.set_counts[src_parent]);
        }

        return true;
    }
};

test "alternate disjoint set: python merge examples" {
    var ds = try DisjointSet.init(testing.allocator, &[_]usize{ 1, 1, 1 });
    defer ds.deinit();

    try testing.expect(try ds.merge(1, 2));
    try testing.expect(try ds.merge(0, 2));
    try testing.expect(!(try ds.merge(0, 1)));

    try testing.expectEqual(@as(usize, 3), ds.max_set);
}

test "alternate disjoint set: python parent examples" {
    var ds = try DisjointSet.init(testing.allocator, &[_]usize{ 1, 1, 1 });
    defer ds.deinit();

    try testing.expect(try ds.merge(1, 2));

    try testing.expectEqual(@as(usize, 0), try ds.getParent(0));
    try testing.expectEqual(@as(usize, 2), try ds.getParent(1));

    try testing.expectError(error.IndexOutOfBounds, ds.getParent(10));
    try testing.expectError(error.IndexOutOfBounds, ds.merge(0, 9));
}

test "alternate disjoint set: extreme chain merges" {
    const n: usize = 100_000;

    const counts = try testing.allocator.alloc(usize, n);
    defer testing.allocator.free(counts);
    @memset(counts, 1);

    var ds = try DisjointSet.init(testing.allocator, counts);
    defer ds.deinit();

    var i: usize = 1;
    while (i < n) : (i += 1) {
        _ = try ds.merge(i - 1, i);
    }

    const root = try ds.getParent(0);
    i = 1;
    while (i < n) : (i += 1) {
        try testing.expectEqual(root, try ds.getParent(i));
    }

    try testing.expectEqual(n, ds.max_set);
}
