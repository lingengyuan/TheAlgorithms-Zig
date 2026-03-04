//! Generic Heap (Item + Score Mapping) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/heap/heap_generic.py

const std = @import("std");
const testing = std.testing;

pub const Mode = enum {
    max,
    min,
};

pub const Entry = struct {
    item: i64,
    score: i64,
};

pub const GenericHeap = struct {
    arr: std.ArrayListUnmanaged(Entry) = .{},
    pos_map: std.AutoHashMap(i64, usize),
    size: usize = 0,
    mode: Mode,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, mode: Mode) GenericHeap {
        return .{
            .pos_map = std.AutoHashMap(i64, usize).init(allocator),
            .mode = mode,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GenericHeap) void {
        self.arr.deinit(self.allocator);
        self.pos_map.deinit();
        self.size = 0;
    }

    fn toScore(self: *const GenericHeap, value: i64) i64 {
        return switch (self.mode) {
            .max => value,
            .min => -value,
        };
    }

    fn parent(i: usize) ?usize {
        return if (i > 0) (i - 1) / 2 else null;
    }

    fn left(self: *const GenericHeap, i: usize) ?usize {
        const l = 2 * i + 1;
        return if (l < self.size) l else null;
    }

    fn right(self: *const GenericHeap, i: usize) ?usize {
        const r = 2 * i + 2;
        return if (r < self.size) r else null;
    }

    fn cmp(self: *const GenericHeap, i: usize, j: usize) bool {
        return self.arr.items[i].score < self.arr.items[j].score;
    }

    fn swap(self: *GenericHeap, i: usize, j: usize) !void {
        std.mem.swap(Entry, &self.arr.items[i], &self.arr.items[j]);
        try self.pos_map.put(self.arr.items[i].item, i);
        try self.pos_map.put(self.arr.items[j].item, j);
    }

    fn getValidParent(self: *const GenericHeap, i: usize) usize {
        var valid = i;
        if (self.left(i)) |l| {
            if (!self.cmp(l, valid)) valid = l;
        }
        if (self.right(i)) |r| {
            if (!self.cmp(r, valid)) valid = r;
        }
        return valid;
    }

    fn heapifyUp(self: *GenericHeap, index_in: usize) !void {
        var index = index_in;
        while (parent(index)) |p| {
            if (self.cmp(index, p)) break;
            try self.swap(index, p);
            index = p;
        }
    }

    fn heapifyDown(self: *GenericHeap, index_in: usize) !void {
        var index = index_in;
        var valid = self.getValidParent(index);
        while (valid != index) {
            try self.swap(index, valid);
            index = valid;
            valid = self.getValidParent(index);
        }
    }

    /// Updates existing item value if present.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn updateItem(self: *GenericHeap, item: i64, item_value: i64) !void {
        const index = self.pos_map.get(item) orelse return;
        self.arr.items[index] = .{ .item = item, .score = self.toScore(item_value) };
        try self.heapifyUp(index);
        try self.heapifyDown(index);
    }

    /// Deletes item if present.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn deleteItem(self: *GenericHeap, item: i64) !void {
        const removed = self.pos_map.fetchRemove(item) orelse return;
        const index = removed.value;
        const last_index = self.size - 1;

        if (index != last_index) {
            self.arr.items[index] = self.arr.items[last_index];
            try self.pos_map.put(self.arr.items[index].item, index);
        }

        _ = self.arr.pop();
        self.size -= 1;

        if (self.size > index) {
            try self.heapifyUp(index);
            try self.heapifyDown(index);
        }
    }

    /// Inserts item with item_value.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn insertItem(self: *GenericHeap, item: i64, item_value: i64) !void {
        if (self.pos_map.contains(item)) {
            try self.updateItem(item, item_value);
            return;
        }

        const entry = Entry{ .item = item, .score = self.toScore(item_value) };
        if (self.arr.items.len == self.size) {
            try self.arr.append(self.allocator, entry);
        } else {
            self.arr.items[self.size] = entry;
        }
        try self.pos_map.put(item, self.size);
        self.size += 1;
        try self.heapifyUp(self.size - 1);
    }

    pub fn getTop(self: *const GenericHeap) ?Entry {
        return if (self.size == 0) null else self.arr.items[0];
    }

    /// Extracts top entry if present.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn extractTop(self: *GenericHeap) !?Entry {
        const top = self.getTop() orelse return null;
        try self.deleteItem(top.item);
        return top;
    }
};

test "heap generic: python max heap examples" {
    var h = GenericHeap.init(testing.allocator, .max);
    defer h.deinit();

    try h.insertItem(5, 34);
    try h.insertItem(6, 31);
    try h.insertItem(7, 37);

    try testing.expectEqual(Entry{ .item = 7, .score = 37 }, h.getTop().?);
    try testing.expectEqual(Entry{ .item = 7, .score = 37 }, (try h.extractTop()).?);
    try testing.expectEqual(Entry{ .item = 5, .score = 34 }, (try h.extractTop()).?);
    try testing.expectEqual(Entry{ .item = 6, .score = 31 }, (try h.extractTop()).?);
}

test "heap generic: python min heap style examples" {
    var h = GenericHeap.init(testing.allocator, .min);
    defer h.deinit();

    try h.insertItem(5, 34);
    try h.insertItem(6, 31);
    try h.insertItem(7, 37);

    try testing.expectEqual(Entry{ .item = 6, .score = -31 }, h.getTop().?);
    try testing.expectEqual(Entry{ .item = 6, .score = -31 }, (try h.extractTop()).?);
    try testing.expectEqual(Entry{ .item = 5, .score = -34 }, (try h.extractTop()).?);
    try testing.expectEqual(Entry{ .item = 7, .score = -37 }, (try h.extractTop()).?);

    try h.insertItem(8, 45);
    try h.insertItem(9, 40);
    try h.insertItem(10, 50);
    try testing.expectEqual(@as(i64, 9), h.getTop().?.item);

    try h.updateItem(10, 30);
    try testing.expectEqual(@as(i64, 10), h.getTop().?.item);

    try h.deleteItem(10);
    try testing.expectEqual(@as(i64, 9), h.getTop().?.item);
}

test "heap generic: extreme randomized extraction order" {
    const n: usize = 30_000;

    var h = GenericHeap.init(testing.allocator, .max);
    defer h.deinit();

    var prng = std.Random.DefaultPrng.init(0x99887766);
    const random = prng.random();

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const value: i64 = @intCast(random.intRangeAtMost(i32, -1_000_000, 1_000_000));
        try h.insertItem(@intCast(i), value);
    }

    i = 0;
    while (i < 10_000) : (i += 1) {
        const item = random.uintLessThan(usize, n);
        const value: i64 = @intCast(random.intRangeAtMost(i32, -1_000_000, 1_000_000));
        try h.updateItem(@intCast(item), value);
    }

    var prev = (try h.extractTop()).?.score;
    var count: usize = 1;
    while (try h.extractTop()) |entry| : (count += 1) {
        try testing.expect(prev >= entry.score);
        prev = entry.score;
    }

    try testing.expectEqual(n, count);
}
