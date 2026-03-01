//! Fenwick Tree (Binary Indexed Tree) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/fenwick_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const FenwickTree = struct {
    const Self = @This();

    allocator: Allocator,
    n: usize,
    tree: []i64, // 1-based internal indexing, length n + 1

    pub fn init(allocator: Allocator, size: usize) !Self {
        const tree = try allocator.alloc(i64, size + 1);
        @memset(tree, 0);
        return .{
            .allocator = allocator,
            .n = size,
            .tree = tree,
        };
    }

    pub fn fromSlice(allocator: Allocator, values: []const i64) !Self {
        var fw = try Self.init(allocator, values.len);
        for (values, 0..) |v, i| {
            try fw.add(i, v);
        }
        return fw;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.tree);
        self.* = undefined;
    }

    pub fn len(self: *const Self) usize {
        return self.n;
    }

    pub fn add(self: *Self, index: usize, delta: i64) !void {
        if (index >= self.n) return error.IndexOutOfBounds;
        var i = index + 1;
        while (i <= self.n) : (i += lowbit(i)) {
            self.tree[i] +%= delta;
        }
    }

    pub fn set(self: *Self, index: usize, value: i64) !void {
        const cur = try self.get(index);
        try self.add(index, value - cur);
    }

    pub fn prefixSum(self: *const Self, right: usize) !i64 {
        // Sum of [0, right)
        if (right > self.n) return error.IndexOutOfBounds;
        var sum: i64 = 0;
        var i = right;
        while (i > 0) : (i -= lowbit(i)) {
            sum +%= self.tree[i];
        }
        return sum;
    }

    pub fn rangeSum(self: *const Self, left: usize, right: usize) !i64 {
        if (left > right or right > self.n) return error.InvalidRange;
        const r = try self.prefixSum(right);
        const l = try self.prefixSum(left);
        return r - l;
    }

    pub fn get(self: *const Self, index: usize) !i64 {
        if (index >= self.n) return error.IndexOutOfBounds;
        return try self.rangeSum(index, index + 1);
    }

    fn lowbit(x: usize) usize {
        return x & (~x +% 1);
    }
};

test "fenwick tree: build and prefix/range sums" {
    var fw = try FenwickTree.fromSlice(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 });
    defer fw.deinit();

    try testing.expectEqual(@as(i64, 0), try fw.prefixSum(0));
    try testing.expectEqual(@as(i64, 6), try fw.prefixSum(3));
    try testing.expectEqual(@as(i64, 15), try fw.prefixSum(5));
    try testing.expectEqual(@as(i64, 9), try fw.rangeSum(1, 4));
}

test "fenwick tree: add updates" {
    var fw = try FenwickTree.fromSlice(testing.allocator, &[_]i64{ 1, 1, 1, 1 });
    defer fw.deinit();

    try fw.add(0, 2);
    try fw.add(3, 5);
    try testing.expectEqual(@as(i64, 3), try fw.get(0));
    try testing.expectEqual(@as(i64, 6), try fw.get(3));
    try testing.expectEqual(@as(i64, 11), try fw.prefixSum(4));
}

test "fenwick tree: set operation" {
    var fw = try FenwickTree.fromSlice(testing.allocator, &[_]i64{ 5, 4, 3, 2, 1 });
    defer fw.deinit();

    try fw.set(0, 1);
    try fw.set(4, 5);
    try testing.expectEqual(@as(i64, 1), try fw.get(0));
    try testing.expectEqual(@as(i64, 5), try fw.get(4));
    try testing.expectEqual(@as(i64, 15), try fw.prefixSum(5));
}

test "fenwick tree: empty tree behaves" {
    var fw = try FenwickTree.init(testing.allocator, 0);
    defer fw.deinit();

    try testing.expectEqual(@as(i64, 0), try fw.prefixSum(0));
    try testing.expectError(error.IndexOutOfBounds, fw.add(0, 1));
    try testing.expectError(error.IndexOutOfBounds, fw.get(0));
}

test "fenwick tree: invalid ranges and indices" {
    var fw = try FenwickTree.fromSlice(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer fw.deinit();

    try testing.expectError(error.IndexOutOfBounds, fw.prefixSum(4));
    try testing.expectError(error.InvalidRange, fw.rangeSum(2, 1));
    try testing.expectError(error.InvalidRange, fw.rangeSum(0, 4));
    try testing.expectError(error.IndexOutOfBounds, fw.set(3, 9));
}

test "fenwick tree: supports negative values" {
    var fw = try FenwickTree.fromSlice(testing.allocator, &[_]i64{ -3, 1, -2, 5 });
    defer fw.deinit();

    try testing.expectEqual(@as(i64, 1), try fw.prefixSum(4));
    try fw.add(2, 4);
    try testing.expectEqual(@as(i64, 3), try fw.rangeSum(1, 3));
}
