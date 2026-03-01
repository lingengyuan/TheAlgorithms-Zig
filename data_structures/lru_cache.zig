//! LRU Cache - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/other/lru_cache.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LruCache = struct {
    const Self = @This();

    const Node = struct {
        key: i64,
        value: i64,
        prev: ?*Node,
        next: ?*Node,
    };

    allocator: Allocator,
    capacity: usize,
    size: usize,
    hits: usize,
    misses: usize,
    map: std.AutoHashMap(i64, *Node),
    head: *Node, // sentinel LRU side
    tail: *Node, // sentinel MRU side

    pub fn init(allocator: Allocator, capacity: usize) !Self {
        const head = try allocator.create(Node);
        const tail = try allocator.create(Node);
        head.* = .{ .key = 0, .value = 0, .prev = null, .next = tail };
        tail.* = .{ .key = 0, .value = 0, .prev = head, .next = null };

        return .{
            .allocator = allocator,
            .capacity = capacity,
            .size = 0,
            .hits = 0,
            .misses = 0,
            .map = std.AutoHashMap(i64, *Node).init(allocator),
            .head = head,
            .tail = tail,
        };
    }

    pub fn deinit(self: *Self) void {
        var cur = self.head.next;
        while (cur != null and cur.? != self.tail) {
            const node = cur.?;
            cur = node.next;
            self.allocator.destroy(node);
        }
        self.allocator.destroy(self.head);
        self.allocator.destroy(self.tail);
        self.map.deinit();
        self.* = undefined;
    }

    pub fn len(self: *const Self) usize {
        return self.size;
    }

    pub fn get(self: *Self, key: i64) ?i64 {
        const node = self.map.get(key) orelse {
            self.misses += 1;
            return null;
        };

        self.hits += 1;
        self.moveToBack(node);
        return node.value;
    }

    pub fn put(self: *Self, key: i64, value: i64) !void {
        if (self.capacity == 0) return;

        if (self.map.get(key)) |node| {
            node.value = value;
            self.moveToBack(node);
            return;
        }

        const node = try self.allocator.create(Node);
        errdefer self.allocator.destroy(node);
        node.* = .{ .key = key, .value = value, .prev = null, .next = null };

        try self.map.put(key, node);

        if (self.size >= self.capacity) {
            self.evictFront();
        }
        self.insertBeforeTail(node);
        self.size += 1;
    }

    pub fn cacheInfo(self: *const Self) struct { hits: usize, misses: usize, capacity: usize, size: usize } {
        return .{
            .hits = self.hits,
            .misses = self.misses,
            .capacity = self.capacity,
            .size = self.size,
        };
    }

    fn evictFront(self: *Self) void {
        const node = self.head.next orelse return;
        if (node == self.tail) return;

        self.detach(node);
        _ = self.map.remove(node.key);
        self.allocator.destroy(node);
        self.size -= 1;
    }

    fn moveToBack(self: *Self, node: *Node) void {
        self.detach(node);
        self.insertBeforeTail(node);
    }

    fn detach(self: *Self, node: *Node) void {
        _ = self;
        const p = node.prev orelse return;
        const n = node.next orelse return;
        p.next = n;
        n.prev = p;
        node.prev = null;
        node.next = null;
    }

    fn insertBeforeTail(self: *Self, node: *Node) void {
        const prev = self.tail.prev orelse self.head;
        prev.next = node;
        node.prev = prev;
        node.next = self.tail;
        self.tail.prev = node;
    }
};

test "lru cache: basic get/put and eviction" {
    var cache = try LruCache.init(testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try testing.expectEqual(@as(?i64, 10), cache.get(1)); // 2 is now LRU
    try cache.put(3, 30); // evict 2

    try testing.expectEqual(@as(?i64, null), cache.get(2));
    try testing.expectEqual(@as(?i64, 10), cache.get(1));
    try testing.expectEqual(@as(?i64, 30), cache.get(3));
}

test "lru cache: update existing key moves to most recent" {
    var cache = try LruCache.init(testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 1);
    try cache.put(2, 2);
    try cache.put(1, 11); // update and move 1 to MRU
    try cache.put(3, 3); // should evict key 2

    try testing.expectEqual(@as(?i64, 11), cache.get(1));
    try testing.expectEqual(@as(?i64, null), cache.get(2));
    try testing.expectEqual(@as(?i64, 3), cache.get(3));
}

test "lru cache: capacity zero stores nothing" {
    var cache = try LruCache.init(testing.allocator, 0);
    defer cache.deinit();

    try cache.put(1, 1);
    try cache.put(2, 2);
    try testing.expectEqual(@as(?i64, null), cache.get(1));
    try testing.expectEqual(@as(usize, 0), cache.len());
}

test "lru cache: single-capacity behavior" {
    var cache = try LruCache.init(testing.allocator, 1);
    defer cache.deinit();

    try cache.put(7, 70);
    try testing.expectEqual(@as(?i64, 70), cache.get(7));
    try cache.put(8, 80);
    try testing.expectEqual(@as(?i64, null), cache.get(7));
    try testing.expectEqual(@as(?i64, 80), cache.get(8));
}

test "lru cache: hit/miss accounting" {
    var cache = try LruCache.init(testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 1);
    _ = cache.get(1); // hit
    _ = cache.get(2); // miss
    _ = cache.get(1); // hit
    const info = cache.cacheInfo();
    try testing.expectEqual(@as(usize, 2), info.hits);
    try testing.expectEqual(@as(usize, 1), info.misses);
    try testing.expectEqual(@as(usize, 1), info.size);
}

test "lru cache: failed map insertion does not corrupt state" {
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 3 });
    const alloc = failing.allocator();

    var cache = try LruCache.init(alloc, 2);
    defer cache.deinit();

    try testing.expectError(error.OutOfMemory, cache.put(1, 1));
    try testing.expectEqual(@as(usize, 0), cache.len());
    try testing.expectEqual(@as(?i64, null), cache.get(1));
}
