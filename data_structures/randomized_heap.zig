//! Randomized Meldable Heap - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/heap/randomized_heap.py

const std = @import("std");
const testing = std.testing;

pub const RandomizedHeapNode = struct {
    value: i64,
    left: ?*RandomizedHeapNode = null,
    right: ?*RandomizedHeapNode = null,
};

fn nextRand(state: *u64) u64 {
    var x = state.*;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state.* = x;
    return x;
}

fn randomBool(state: *u64) bool {
    return (nextRand(state) & 1) == 1;
}

fn mergeNodes(
    root1_in: ?*RandomizedHeapNode,
    root2_in: ?*RandomizedHeapNode,
    rng_state: *u64,
) ?*RandomizedHeapNode {
    var root1 = root1_in;
    var root2 = root2_in;

    if (root1 == null) return root2;
    if (root2 == null) return root1;

    if (root1.?.value > root2.?.value) {
        std.mem.swap(?*RandomizedHeapNode, &root1, &root2);
    }

    if (randomBool(rng_state)) {
        std.mem.swap(?*RandomizedHeapNode, &root1.?.left, &root1.?.right);
    }

    root1.?.left = mergeNodes(root1.?.left, root2, rng_state);
    return root1;
}

pub const RandomizedHeap = struct {
    root: ?*RandomizedHeapNode = null,
    allocator: std.mem.Allocator,
    rng_state: u64 = 0xC0FFEE123456789,

    pub fn init(allocator: std.mem.Allocator) RandomizedHeap {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RandomizedHeap) void {
        const start = self.root orelse return;

        var stack = std.ArrayListUnmanaged(*RandomizedHeapNode){};
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

    pub fn isEmpty(self: *const RandomizedHeap) bool {
        return self.root == null;
    }

    /// Inserts one value.
    /// Time complexity: expected O(log n), Space complexity: expected O(log n)
    pub fn insert(self: *RandomizedHeap, value: i64) !void {
        const node = try self.allocator.create(RandomizedHeapNode);
        node.* = .{ .value = value };
        self.root = mergeNodes(self.root, node, &self.rng_state);
    }

    /// Returns smallest value.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn top(self: *const RandomizedHeap) !i64 {
        const r = self.root orelse return error.EmptyHeap;
        return r.value;
    }

    /// Pops smallest value.
    /// Time complexity: expected O(log n), Space complexity: expected O(log n)
    pub fn pop(self: *RandomizedHeap) !i64 {
        const r = self.root orelse return error.EmptyHeap;
        const value = r.value;

        const left = r.left;
        const right = r.right;
        self.allocator.destroy(r);

        self.root = mergeNodes(left, right, &self.rng_state);
        return value;
    }

    pub fn clear(self: *RandomizedHeap) void {
        self.deinit();
    }

    pub fn toSortedList(self: *RandomizedHeap, allocator: std.mem.Allocator) ![]i64 {
        var out = std.ArrayListUnmanaged(i64){};
        errdefer out.deinit(allocator);

        while (!self.isEmpty()) {
            try out.append(allocator, try self.pop());
        }

        return out.toOwnedSlice(allocator);
    }
};

test "randomized heap: python to_sorted_list examples" {
    var rh = RandomizedHeap.init(testing.allocator);
    defer rh.deinit();

    for ([_]i64{ 2, 3, 1, 5, 1, 7 }) |v| {
        try rh.insert(v);
    }

    const sorted = try rh.toSortedList(testing.allocator);
    defer testing.allocator.free(sorted);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 2, 3, 5, 7 }, sorted);
}

test "randomized heap: pop/top/clear boundaries" {
    var rh = RandomizedHeap.init(testing.allocator);
    defer rh.deinit();

    try testing.expectError(error.EmptyHeap, rh.pop());

    try rh.insert(3);
    try testing.expectEqual(@as(i64, 3), try rh.top());
    try rh.insert(1);
    try testing.expectEqual(@as(i64, 1), try rh.top());
    try rh.insert(3);
    try rh.insert(7);

    try testing.expectEqual(@as(i64, 1), try rh.pop());
    try testing.expectEqual(@as(i64, 3), try rh.pop());
    try testing.expectEqual(@as(i64, 3), try rh.pop());
    try testing.expectEqual(@as(i64, 7), try rh.pop());

    try testing.expectError(error.EmptyHeap, rh.pop());

    for ([_]i64{ 9, 8, 7 }) |v| try rh.insert(v);
    rh.clear();
    try testing.expectError(error.EmptyHeap, rh.top());
}

test "randomized heap: extreme randomized ordering" {
    const n: usize = 30_000;

    var rh = RandomizedHeap.init(testing.allocator);
    defer rh.deinit();

    var prng = std.Random.DefaultPrng.init(0xFEEDBEEF);
    const random = prng.random();

    for (0..n) |_| {
        const v: i64 = @intCast(random.intRangeAtMost(i32, -700_000, 700_000));
        try rh.insert(v);
    }

    var prev = try rh.pop();
    var count: usize = 1;
    while (!rh.isEmpty()) : (count += 1) {
        const cur = try rh.pop();
        try testing.expect(prev <= cur);
        prev = cur;
    }

    try testing.expectEqual(n, count);
}
