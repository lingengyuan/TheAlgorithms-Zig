//! Priority Queue (list-backed binary heap) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/priority_queue_using_list.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Min-priority queue: smaller `priority` is dequeued first.
/// If priority ties, smaller `value` is dequeued first.
pub const PriorityQueue = struct {
    const Self = @This();

    pub const Item = struct {
        value: i64,
        priority: i64,
    };

    allocator: Allocator,
    items: std.ArrayListUnmanaged(Item) = .{},

    pub fn init(allocator: Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self.items.deinit(self.allocator);
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.items.items.len == 0;
    }

    pub fn size(self: *const Self) usize {
        return self.items.items.len;
    }

    pub fn enqueue(self: *Self, value: i64, priority: i64) !void {
        try self.items.append(self.allocator, .{ .value = value, .priority = priority });
        self.siftUp(self.items.items.len - 1);
    }

    pub fn peek(self: *const Self) ?Item {
        if (self.items.items.len == 0) return null;
        return self.items.items[0];
    }

    pub fn dequeue(self: *Self) ?Item {
        const len = self.items.items.len;
        if (len == 0) return null;

        const out = self.items.items[0];
        self.items.items[0] = self.items.items[len - 1];
        _ = self.items.pop();
        if (self.items.items.len > 0) {
            self.siftDown(0);
        }
        return out;
    }

    fn less(a: Item, b: Item) bool {
        if (a.priority != b.priority) return a.priority < b.priority;
        return a.value < b.value;
    }

    fn siftUp(self: *Self, idx_: usize) void {
        var idx = idx_;
        while (idx > 0) {
            const p = (idx - 1) / 2;
            if (less(self.items.items[idx], self.items.items[p])) {
                std.mem.swap(Item, &self.items.items[idx], &self.items.items[p]);
                idx = p;
            } else break;
        }
    }

    fn siftDown(self: *Self, idx_: usize) void {
        var idx = idx_;
        const len = self.items.items.len;
        while (true) {
            var smallest = idx;
            const left = 2 * idx + 1;
            const right = 2 * idx + 2;

            if (left < len and less(self.items.items[left], self.items.items[smallest])) {
                smallest = left;
            }
            if (right < len and less(self.items.items[right], self.items.items[smallest])) {
                smallest = right;
            }
            if (smallest != idx) {
                std.mem.swap(Item, &self.items.items[idx], &self.items.items[smallest]);
                idx = smallest;
            } else break;
        }
    }
};

test "priority queue: enqueue/dequeue ordering" {
    var pq = PriorityQueue.init(testing.allocator);
    defer pq.deinit();

    try pq.enqueue(100, 3);
    try pq.enqueue(200, 1);
    try pq.enqueue(150, 2);
    try pq.enqueue(120, 1);

    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 120, .priority = 1 }), pq.dequeue());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 200, .priority = 1 }), pq.dequeue());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 150, .priority = 2 }), pq.dequeue());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 100, .priority = 3 }), pq.dequeue());
    try testing.expectEqual(@as(?PriorityQueue.Item, null), pq.dequeue());
}

test "priority queue: peek and size" {
    var pq = PriorityQueue.init(testing.allocator);
    defer pq.deinit();

    try testing.expect(pq.isEmpty());
    try testing.expectEqual(@as(?PriorityQueue.Item, null), pq.peek());

    try pq.enqueue(5, 9);
    try pq.enqueue(7, 8);
    try testing.expectEqual(@as(usize, 2), pq.size());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 7, .priority = 8 }), pq.peek());
}

test "priority queue: duplicates" {
    var pq = PriorityQueue.init(testing.allocator);
    defer pq.deinit();

    try pq.enqueue(4, 2);
    try pq.enqueue(4, 2);
    try pq.enqueue(1, 2);

    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 1, .priority = 2 }), pq.dequeue());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 4, .priority = 2 }), pq.dequeue());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 4, .priority = 2 }), pq.dequeue());
}

test "priority queue: single element" {
    var pq = PriorityQueue.init(testing.allocator);
    defer pq.deinit();

    try pq.enqueue(42, 0);
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 42, .priority = 0 }), pq.peek());
    try testing.expectEqual(@as(?PriorityQueue.Item, .{ .value = 42, .priority = 0 }), pq.dequeue());
    try testing.expect(pq.isEmpty());
}
