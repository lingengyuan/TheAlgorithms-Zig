//! Segment Tree (Recursive Node Form) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/segment_tree_other.py

const std = @import("std");
const testing = std.testing;

pub fn addCombine(a: i64, b: i64) i64 {
    return a + b;
}

pub fn maxCombine(a: i64, b: i64) i64 {
    return @max(a, b);
}

pub fn minCombine(a: i64, b: i64) i64 {
    return @min(a, b);
}

pub const SegmentTreeNode = struct {
    start: usize,
    end: usize,
    val: i64,
    left: ?*SegmentTreeNode = null,
    right: ?*SegmentTreeNode = null,
};

pub const NodeSnapshot = struct {
    start: usize,
    end: usize,
    val: i64,
};

pub const SegmentTree = struct {
    root: ?*SegmentTreeNode = null,
    combine: *const fn (i64, i64) i64,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        collection: []const i64,
        combine: *const fn (i64, i64) i64,
    ) !SegmentTree {
        var tree = SegmentTree{ .combine = combine, .allocator = allocator };
        if (collection.len > 0) {
            tree.root = try tree.buildTree(collection, 0, collection.len - 1);
        }
        return tree;
    }

    pub fn deinit(self: *SegmentTree) void {
        const start = self.root orelse return;
        var stack = std.ArrayListUnmanaged(*SegmentTreeNode){};
        defer stack.deinit(self.allocator);
        stack.append(self.allocator, start) catch return;

        while (stack.items.len > 0) {
            const node = stack.pop().?;
            if (node.left) |left| stack.append(self.allocator, left) catch {};
            if (node.right) |right| stack.append(self.allocator, right) catch {};
            self.allocator.destroy(node);
        }

        self.root = null;
    }

    fn buildTree(self: *SegmentTree, collection: []const i64, start: usize, end: usize) !*SegmentTreeNode {
        const node = try self.allocator.create(SegmentTreeNode);
        errdefer self.allocator.destroy(node);

        if (start == end) {
            node.* = .{ .start = start, .end = end, .val = collection[start] };
            return node;
        }

        const mid = (start + end) / 2;
        const left = try self.buildTree(collection, start, mid);
        errdefer self.deinitNode(left);
        const right = try self.buildTree(collection, mid + 1, end);
        errdefer self.deinitNode(right);

        node.* = .{
            .start = start,
            .end = end,
            .val = self.combine(left.val, right.val),
            .left = left,
            .right = right,
        };

        return node;
    }

    fn deinitNode(self: *SegmentTree, root: *SegmentTreeNode) void {
        var stack = std.ArrayListUnmanaged(*SegmentTreeNode){};
        defer stack.deinit(self.allocator);
        stack.append(self.allocator, root) catch return;

        while (stack.items.len > 0) {
            const node = stack.pop().?;
            if (node.left) |left| stack.append(self.allocator, left) catch {};
            if (node.right) |right| stack.append(self.allocator, right) catch {};
            self.allocator.destroy(node);
        }
    }

    /// Updates one element.
    /// Time complexity: O(log n), Space complexity: O(log n)
    pub fn update(self: *SegmentTree, i: usize, val: i64) !void {
        const root = self.root orelse return error.EmptyTree;
        if (i > root.end) return error.IndexOutOfBounds;
        self.updateTree(root, i, val);
    }

    fn updateTree(self: *SegmentTree, node: *SegmentTreeNode, i: usize, val: i64) void {
        if (node.start == i and node.end == i) {
            node.val = val;
            return;
        }

        const mid = (node.start + node.end) / 2;
        if (i <= mid) {
            self.updateTree(node.left.?, i, val);
        } else {
            self.updateTree(node.right.?, i, val);
        }

        node.val = self.combine(node.left.?.val, node.right.?.val);
    }

    /// Queries inclusive range [i, j].
    /// Time complexity: O(log n), Space complexity: O(log n)
    pub fn queryRange(self: *const SegmentTree, i: usize, j: usize) !i64 {
        const root = self.root orelse return error.EmptyTree;
        if (i > j or j > root.end) return error.InvalidRange;
        return self.queryTree(root, i, j);
    }

    fn queryTree(self: *const SegmentTree, node: *const SegmentTreeNode, i: usize, j: usize) i64 {
        if (node.start == i and node.end == j) {
            return node.val;
        }

        const mid = (node.start + node.end) / 2;
        if (i <= mid) {
            if (j <= mid) {
                return self.queryTree(node.left.?, i, j);
            }

            return self.combine(
                self.queryTree(node.left.?, i, mid),
                self.queryTree(node.right.?, mid + 1, j),
            );
        }

        return self.queryTree(node.right.?, i, j);
    }

    pub fn traverseLevelOrder(self: *const SegmentTree, allocator: std.mem.Allocator) ![]NodeSnapshot {
        var out = std.ArrayListUnmanaged(NodeSnapshot){};
        errdefer out.deinit(allocator);

        const start = self.root orelse return out.toOwnedSlice(allocator);

        var queue = std.ArrayListUnmanaged(*const SegmentTreeNode){};
        defer queue.deinit(allocator);
        try queue.append(allocator, start);

        var head: usize = 0;
        while (head < queue.items.len) {
            const node = queue.items[head];
            head += 1;

            try out.append(allocator, .{ .start = node.start, .end = node.end, .val = node.val });
            if (node.left) |left| try queue.append(allocator, left);
            if (node.right) |right| try queue.append(allocator, right);
        }

        return out.toOwnedSlice(allocator);
    }
};

fn naiveRangeSum(values: []const i64, i: usize, j: usize) i64 {
    var total: i64 = 0;
    for (values[i .. j + 1]) |v| total += v;
    return total;
}

test "segment tree other: python doctest style with add" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{ 2, 1, 5, 3, 4 }, addCombine);
    defer st.deinit();

    const before = try st.traverseLevelOrder(testing.allocator);
    defer testing.allocator.free(before);

    try testing.expectEqual(@as(usize, 9), before.len);
    try testing.expectEqual(NodeSnapshot{ .start = 0, .end = 4, .val = 15 }, before[0]);
    try testing.expectEqual(NodeSnapshot{ .start = 0, .end = 2, .val = 8 }, before[1]);
    try testing.expectEqual(NodeSnapshot{ .start = 3, .end = 4, .val = 7 }, before[2]);

    try st.update(1, 5);

    const after = try st.traverseLevelOrder(testing.allocator);
    defer testing.allocator.free(after);

    try testing.expectEqual(NodeSnapshot{ .start = 0, .end = 4, .val = 19 }, after[0]);
    try testing.expectEqual(@as(i64, 7), try st.queryRange(3, 4));
    try testing.expectEqual(@as(i64, 5), try st.queryRange(2, 2));
    try testing.expectEqual(@as(i64, 13), try st.queryRange(1, 3));
}

test "segment tree other: max/min parity" {
    var max_st = try SegmentTree.init(testing.allocator, &[_]i64{ 2, 1, 5, 3, 4 }, maxCombine);
    defer max_st.deinit();
    try testing.expectEqual(@as(i64, 5), try max_st.queryRange(1, 3));
    try max_st.update(1, 5);
    try testing.expectEqual(@as(i64, 4), try max_st.queryRange(3, 4));

    var min_st = try SegmentTree.init(testing.allocator, &[_]i64{ 2, 1, 5, 3, 4 }, minCombine);
    defer min_st.deinit();
    try testing.expectEqual(@as(i64, 1), try min_st.queryRange(1, 3));
    try min_st.update(1, 5);
    try testing.expectEqual(@as(i64, 2), try min_st.queryRange(0, 4));
}

test "segment tree other: boundary and extreme randomized" {
    var empty = try SegmentTree.init(testing.allocator, &[_]i64{}, addCombine);
    defer empty.deinit();
    try testing.expectError(error.EmptyTree, empty.queryRange(0, 0));
    try testing.expectError(error.EmptyTree, empty.update(0, 1));

    const n: usize = 15_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    var prng = std.Random.DefaultPrng.init(0x123456789);
    const random = prng.random();

    for (0..n) |i| {
        values[i] = @intCast(random.intRangeAtMost(i32, -3000, 3000));
    }

    var st = try SegmentTree.init(testing.allocator, values, addCombine);
    defer st.deinit();

    var u: usize = 0;
    while (u < 5_000) : (u += 1) {
        const idx = random.uintLessThan(usize, n);
        const val: i64 = @intCast(random.intRangeAtMost(i32, -3000, 3000));
        values[idx] = val;
        try st.update(idx, val);
    }

    var q: usize = 0;
    while (q < 500) : (q += 1) {
        const a = random.uintLessThan(usize, n);
        const b = random.uintLessThan(usize, n);
        const left = @min(a, b);
        const right = @max(a, b);

        try testing.expectEqual(naiveRangeSum(values, left, right), try st.queryRange(left, right));
    }

    try testing.expectError(error.InvalidRange, st.queryRange(4, 3));
    try testing.expectError(error.InvalidRange, st.queryRange(0, n));
    try testing.expectError(error.IndexOutOfBounds, st.update(n, 0));
}
