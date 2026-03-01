//! Segment Tree (Range Max Query) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/segment_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SegmentTree = struct {
    const Self = @This();

    allocator: Allocator,
    n: usize,
    tree: []i64,

    pub fn init(allocator: Allocator, values: []const i64) !Self {
        const n = values.len;
        const size = if (n == 0) @as(usize, 0) else blk: {
            const mul = @mulWithOverflow(n, @as(usize, 4));
            if (mul[1] != 0) return error.Overflow;
            break :blk mul[0];
        };
        const tree = try allocator.alloc(i64, size);
        if (size > 0) @memset(tree, 0);

        var self = Self{
            .allocator = allocator,
            .n = n,
            .tree = tree,
        };
        if (n > 0) {
            self.build(1, 0, n - 1, values);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.tree);
        self.* = undefined;
    }

    pub fn len(self: *const Self) usize {
        return self.n;
    }

    pub fn query(self: *const Self, left: usize, right: usize) !i64 {
        if (self.n == 0) return error.EmptyTree;
        if (left > right or right >= self.n) return error.InvalidRange;
        return self.queryRec(1, 0, self.n - 1, left, right);
    }

    pub fn update(self: *Self, index: usize, value: i64) !void {
        if (self.n == 0) return error.EmptyTree;
        if (index >= self.n) return error.IndexOutOfBounds;
        self.updateRec(1, 0, self.n - 1, index, value);
    }

    fn build(self: *Self, node: usize, left: usize, right: usize, values: []const i64) void {
        if (left == right) {
            self.tree[node] = values[left];
            return;
        }

        const mid = left + (right - left) / 2;
        self.build(node * 2, left, mid, values);
        self.build(node * 2 + 1, mid + 1, right, values);
        self.tree[node] = @max(self.tree[node * 2], self.tree[node * 2 + 1]);
    }

    fn updateRec(self: *Self, node: usize, left: usize, right: usize, index: usize, value: i64) void {
        if (left == right) {
            self.tree[node] = value;
            return;
        }

        const mid = left + (right - left) / 2;
        if (index <= mid) {
            self.updateRec(node * 2, left, mid, index, value);
        } else {
            self.updateRec(node * 2 + 1, mid + 1, right, index, value);
        }
        self.tree[node] = @max(self.tree[node * 2], self.tree[node * 2 + 1]);
    }

    fn queryRec(self: *const Self, node: usize, left: usize, right: usize, ql: usize, qr: usize) i64 {
        if (qr < left or right < ql) return std.math.minInt(i64);
        if (ql <= left and right <= qr) return self.tree[node];

        const mid = left + (right - left) / 2;
        const lv = self.queryRec(node * 2, left, mid, ql, qr);
        const rv = self.queryRec(node * 2 + 1, mid + 1, right, ql, qr);
        return @max(lv, rv);
    }
};

test "segment tree: basic range max queries" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{ 1, 2, -4, 7, 3, -5, 6 });
    defer st.deinit();

    try testing.expectEqual(@as(i64, 7), try st.query(0, 6));
    try testing.expectEqual(@as(i64, 7), try st.query(2, 4));
    try testing.expectEqual(@as(i64, 3), try st.query(4, 4));
}

test "segment tree: point updates" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 });
    defer st.deinit();

    try st.update(2, 10);
    try testing.expectEqual(@as(i64, 10), try st.query(0, 4));
    try st.update(4, -7);
    try testing.expectEqual(@as(i64, 10), try st.query(2, 4));
}

test "segment tree: single element" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{42});
    defer st.deinit();

    try testing.expectEqual(@as(i64, 42), try st.query(0, 0));
    try st.update(0, -10);
    try testing.expectEqual(@as(i64, -10), try st.query(0, 0));
}

test "segment tree: invalid ranges and indices" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{ 3, 1, 4 });
    defer st.deinit();

    try testing.expectError(error.InvalidRange, st.query(2, 1));
    try testing.expectError(error.InvalidRange, st.query(0, 3));
    try testing.expectError(error.IndexOutOfBounds, st.update(3, 9));
}

test "segment tree: empty tree rejects queries and updates" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{});
    defer st.deinit();

    try testing.expectError(error.EmptyTree, st.query(0, 0));
    try testing.expectError(error.EmptyTree, st.update(0, 1));
}

test "segment tree: all negative values" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{ -9, -4, -7, -11 });
    defer st.deinit();

    try testing.expectEqual(@as(i64, -4), try st.query(0, 3));
    try st.update(2, -2);
    try testing.expectEqual(@as(i64, -2), try st.query(1, 3));
}

test "segment tree: oversize input length reports overflow" {
    const huge_len = @divTrunc(std.math.maxInt(usize), @as(usize, 4)) + 1;
    const fake_ptr: [*]const i64 = @ptrFromInt(@alignOf(i64));
    const fake_values = fake_ptr[0..huge_len];
    try testing.expectError(error.Overflow, SegmentTree.init(testing.allocator, fake_values));
}
