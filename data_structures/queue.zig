//! Queue (array-based circular buffer) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/queue_by_list.py

const std = @import("std");
const testing = std.testing;

/// Generic queue backed by a dynamically-resized circular buffer.
pub fn Queue(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        head: usize,
        tail: usize,
        len: usize,
        capacity: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .data = &[_]T{},
                .head = 0,
                .tail = 0,
                .len = 0,
                .capacity = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.capacity > 0) {
                self.allocator.free(self.data[0..self.capacity]);
            }
            self.* = undefined;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        pub fn count(self: *const Self) usize {
            return self.len;
        }

        pub fn enqueue(self: *Self, value: T) !void {
            try self.ensureCapacity(self.len + 1);
            self.data[self.tail] = value;
            self.tail = (self.tail + 1) % self.capacity;
            self.len += 1;
        }

        pub fn dequeue(self: *Self) ?T {
            if (self.len == 0) return null;
            const value = self.data[self.head];
            self.head = (self.head + 1) % self.capacity;
            self.len -= 1;
            return value;
        }

        pub fn peek(self: *const Self) ?T {
            if (self.len == 0) return null;
            return self.data[self.head];
        }

        fn ensureCapacity(self: *Self, min_capacity: usize) !void {
            if (self.capacity >= min_capacity) return;

            var new_capacity: usize = if (self.capacity == 0) 4 else self.capacity * 2;
            if (new_capacity < min_capacity) new_capacity = min_capacity;

            const new_data = try self.allocator.alloc(T, new_capacity);

            for (0..self.len) |i| {
                if (self.capacity == 0) break;
                const old_idx = (self.head + i) % self.capacity;
                new_data[i] = self.data[old_idx];
            }

            if (self.capacity > 0) {
                self.allocator.free(self.data[0..self.capacity]);
            }

            self.data = new_data;
            self.capacity = new_capacity;
            self.head = 0;
            self.tail = self.len;
        }
    };
}

test "queue: enqueue dequeue and peek" {
    var queue = Queue(i32).init(testing.allocator);
    defer queue.deinit();

    try testing.expect(queue.isEmpty());
    try testing.expectEqual(@as(?i32, null), queue.peek());

    try queue.enqueue(10);
    try queue.enqueue(20);
    try queue.enqueue(30);

    try testing.expectEqual(@as(usize, 3), queue.count());
    try testing.expectEqual(@as(?i32, 10), queue.peek());
    try testing.expectEqual(@as(?i32, 10), queue.dequeue());
    try testing.expectEqual(@as(?i32, 20), queue.dequeue());

    try queue.enqueue(40);
    try testing.expectEqual(@as(?i32, 30), queue.dequeue());
    try testing.expectEqual(@as(?i32, 40), queue.dequeue());
    try testing.expectEqual(@as(?i32, null), queue.dequeue());
    try testing.expect(queue.isEmpty());
}

test "queue: works with generic type" {
    var queue = Queue(bool).init(testing.allocator);
    defer queue.deinit();

    try queue.enqueue(true);
    try queue.enqueue(false);
    try testing.expectEqual(@as(?bool, true), queue.dequeue());
    try testing.expectEqual(@as(?bool, false), queue.dequeue());
}

test "queue: circular buffer growth" {
    var queue = Queue(i32).init(testing.allocator);
    defer queue.deinit();

    for (0..50) |i| {
        try queue.enqueue(@intCast(i));
    }
    for (0..25) |i| {
        try testing.expectEqual(@as(?i32, @intCast(i)), queue.dequeue());
    }
    for (50..120) |i| {
        try queue.enqueue(@intCast(i));
    }

    var expected: i32 = 25;
    while (expected < 120) : (expected += 1) {
        try testing.expectEqual(@as(?i32, expected), queue.dequeue());
    }
    try testing.expect(queue.isEmpty());
}
