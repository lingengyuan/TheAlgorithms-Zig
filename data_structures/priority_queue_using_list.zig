//! Priority Queue Using List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/priority_queue_using_list.py

const std = @import("std");
const testing = std.testing;

pub const FixedPriorityQueue = struct {
    allocator: std.mem.Allocator,
    queues: [3]std.ArrayListUnmanaged(i64),

    pub fn init(allocator: std.mem.Allocator) FixedPriorityQueue {
        return .{
            .allocator = allocator,
            .queues = .{ .{}, .{}, .{} },
        };
    }

    pub fn deinit(self: *FixedPriorityQueue) void {
        for (&self.queues) |*queue| {
            queue.deinit(self.allocator);
        }
        self.* = undefined;
    }

    /// Adds an element to one of three fixed priorities (0, 1, 2).
    /// Time complexity: O(1) amortized, Space complexity: O(1)
    pub fn enqueue(self: *FixedPriorityQueue, priority: usize, data: i64) !void {
        if (priority >= self.queues.len) return error.InvalidPriority;
        if (self.queues[priority].items.len >= 100) return error.MaximumQueueSize;
        try self.queues[priority].append(self.allocator, data);
    }

    /// Removes highest-priority element in FIFO order.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn dequeue(self: *FixedPriorityQueue) !i64 {
        for (&self.queues) |*queue| {
            if (queue.items.len > 0) {
                return queue.orderedRemove(0);
            }
        }
        return error.AllQueuesEmpty;
    }
};

pub const ElementPriorityQueue = struct {
    allocator: std.mem.Allocator,
    queue: std.ArrayListUnmanaged(i64),

    pub fn init(allocator: std.mem.Allocator) ElementPriorityQueue {
        return .{
            .allocator = allocator,
            .queue = .{},
        };
    }

    pub fn deinit(self: *ElementPriorityQueue) void {
        self.queue.deinit(self.allocator);
        self.* = undefined;
    }

    /// Adds an element where the element value itself is priority.
    /// Time complexity: O(1) amortized, Space complexity: O(1)
    pub fn enqueue(self: *ElementPriorityQueue, data: i64) !void {
        if (self.queue.items.len >= 100) return error.MaximumQueueSize;
        try self.queue.append(self.allocator, data);
    }

    /// Removes the minimum element.
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn dequeue(self: *ElementPriorityQueue) !i64 {
        if (self.queue.items.len == 0) return error.EmptyQueue;

        var min_index: usize = 0;
        var min_value = self.queue.items[0];

        for (self.queue.items[1..], 1..) |v, i| {
            if (v < min_value) {
                min_value = v;
                min_index = i;
            }
        }

        _ = self.queue.orderedRemove(min_index);
        return min_value;
    }
};

test "priority queue using list: fixed priority queue python sample" {
    var fpq = FixedPriorityQueue.init(testing.allocator);
    defer fpq.deinit();

    try fpq.enqueue(0, 10);
    try fpq.enqueue(1, 70);
    try fpq.enqueue(0, 100);
    try fpq.enqueue(2, 1);
    try fpq.enqueue(2, 5);
    try fpq.enqueue(1, 7);
    try fpq.enqueue(2, 4);
    try fpq.enqueue(1, 64);
    try fpq.enqueue(0, 128);

    const expected = [_]i64{ 10, 100, 128, 70, 7, 64, 1, 5, 4 };
    for (expected) |v| {
        try testing.expectEqual(v, try fpq.dequeue());
    }
    try testing.expectError(error.AllQueuesEmpty, fpq.dequeue());
}

test "priority queue using list: fixed priority validation and extreme" {
    var fpq = FixedPriorityQueue.init(testing.allocator);
    defer fpq.deinit();

    try testing.expectError(error.InvalidPriority, fpq.enqueue(3, 1));

    for (0..100) |i| {
        try fpq.enqueue(2, @intCast(i));
    }
    try testing.expectError(error.MaximumQueueSize, fpq.enqueue(2, 100));

    for (0..100) |i| {
        try testing.expectEqual(@as(i64, @intCast(i)), try fpq.dequeue());
    }
}

test "priority queue using list: element priority queue python sample" {
    var epq = ElementPriorityQueue.init(testing.allocator);
    defer epq.deinit();

    const input = [_]i64{ 10, 70, 4, 1, 5, 7, 4, 64, 128 };
    for (input) |v| {
        try epq.enqueue(v);
    }

    const expected = [_]i64{ 1, 4, 4, 5, 7, 10, 64, 70, 128 };
    for (expected) |v| {
        try testing.expectEqual(v, try epq.dequeue());
    }

    try testing.expectError(error.EmptyQueue, epq.dequeue());
}
