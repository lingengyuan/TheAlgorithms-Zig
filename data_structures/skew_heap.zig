//! Skew Heap - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/heap/skew_heap.py

const std = @import("std");
const testing = std.testing;

pub const SkewNode = struct {
    value: i64,
    left: ?*SkewNode = null,
    right: ?*SkewNode = null,
};

fn mergeNodes(root1_in: ?*SkewNode, root2_in: ?*SkewNode) ?*SkewNode {
    var root1 = root1_in;
    var root2 = root2_in;

    if (root1 == null) return root2;
    if (root2 == null) return root1;

    if (root1.?.value > root2.?.value) {
        std.mem.swap(?*SkewNode, &root1, &root2);
    }

    const result = root1.?;
    const temp = result.right;
    result.right = result.left;
    result.left = mergeNodes(temp, root2);
    return result;
}

pub const SkewHeap = struct {
    root: ?*SkewNode = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SkewHeap {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SkewHeap) void {
        const start = self.root orelse return;

        var stack = std.ArrayListUnmanaged(*SkewNode){};
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

    pub fn isEmpty(self: *const SkewHeap) bool {
        return self.root == null;
    }

    /// Inserts one value.
    /// Time complexity: amortized O(log n), Space complexity: O(log n)
    pub fn insert(self: *SkewHeap, value: i64) !void {
        const node = try self.allocator.create(SkewNode);
        node.* = .{ .value = value };
        self.root = mergeNodes(self.root, node);
    }

    /// Returns minimum value.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn top(self: *const SkewHeap) !i64 {
        const r = self.root orelse return error.EmptyHeap;
        return r.value;
    }

    /// Pops minimum value.
    /// Time complexity: amortized O(log n), Space complexity: O(log n)
    pub fn pop(self: *SkewHeap) !i64 {
        const root = self.root orelse return error.EmptyHeap;
        const value = root.value;

        const left = root.left;
        const right = root.right;
        self.allocator.destroy(root);

        self.root = mergeNodes(left, right);
        return value;
    }

    pub fn clear(self: *SkewHeap) void {
        self.deinit();
    }

    pub fn toSortedList(self: *SkewHeap, allocator: std.mem.Allocator) ![]i64 {
        var out = std.ArrayListUnmanaged(i64){};
        errdefer out.deinit(allocator);

        while (!self.isEmpty()) {
            try out.append(allocator, try self.pop());
        }

        for (out.items) |v| {
            try self.insert(v);
        }

        return out.toOwnedSlice(allocator);
    }
};

test "skew heap: python basic behavior" {
    var sh = SkewHeap.init(testing.allocator);
    defer sh.deinit();

    for ([_]i64{ 2, 3, 1, 5, 1, 7 }) |v| {
        try sh.insert(v);
    }

    const values = try sh.toSortedList(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 2, 3, 5, 7 }, values);
}

test "skew heap: pop and clear boundaries" {
    var sh = SkewHeap.init(testing.allocator);
    defer sh.deinit();

    try testing.expectError(error.EmptyHeap, sh.pop());

    try sh.insert(1);
    try sh.insert(-1);
    try sh.insert(0);

    try testing.expectEqual(@as(i64, -1), try sh.pop());
    try testing.expectEqual(@as(i64, 0), try sh.pop());
    try testing.expectEqual(@as(i64, 1), try sh.pop());

    try testing.expectError(error.EmptyHeap, sh.top());

    for ([_]i64{ 3, 1, 3, 7 }) |v| try sh.insert(v);
    sh.clear();
    try testing.expectError(error.EmptyHeap, sh.pop());
}

test "skew heap: extreme randomized pop ordering" {
    const n: usize = 30_000;
    var sh = SkewHeap.init(testing.allocator);
    defer sh.deinit();

    var prng = std.Random.DefaultPrng.init(0x55AA33CC);
    const random = prng.random();

    for (0..n) |_| {
        const v: i64 = @intCast(random.intRangeAtMost(i32, -500_000, 500_000));
        try sh.insert(v);
    }

    var prev = try sh.pop();
    var count: usize = 1;
    while (!sh.isEmpty()) : (count += 1) {
        const cur = try sh.pop();
        try testing.expect(prev <= cur);
        prev = cur;
    }

    try testing.expectEqual(n, count);
}
