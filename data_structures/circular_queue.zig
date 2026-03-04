//! Circular Queue (fixed capacity) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/circular_queue.py

const std = @import("std");
const testing = std.testing;

pub fn CircularQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        n: usize,
        array: []?T,
        front: usize,
        rear: usize,
        size: usize,

        pub fn init(allocator: std.mem.Allocator, n: usize) !Self {
            if (n == 0) return error.InvalidCapacity;
            const array = try allocator.alloc(?T, n);
            @memset(array, null);
            return .{
                .allocator = allocator,
                .n = n,
                .array = array,
                .front = 0,
                .rear = 0,
                .size = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.array);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.size;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        pub fn first(self: *const Self) ?T {
            return if (self.isEmpty()) null else self.array[self.front];
        }

        /// Inserts element at queue tail.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn enqueue(self: *Self, data: T) !*Self {
            if (self.size >= self.n) return error.QueueFull;
            self.array[self.rear] = data;
            self.rear = (self.rear + 1) % self.n;
            self.size += 1;
            return self;
        }

        /// Removes and returns queue head.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn dequeue(self: *Self) !T {
            if (self.size == 0) return error.Underflow;

            const value = self.array[self.front].?;
            self.array[self.front] = null;
            self.front = (self.front + 1) % self.n;
            self.size -= 1;
            return value;
        }
    };
}

test "circular queue: python-style operations" {
    var cq = try CircularQueue([]const u8).init(testing.allocator, 5);
    defer cq.deinit();

    try testing.expectEqual(@as(usize, 0), cq.len());
    try testing.expect(cq.isEmpty());
    try testing.expectEqual(@as(?[]const u8, null), cq.first());

    _ = try cq.enqueue("A");
    try testing.expectEqualStrings("A", cq.array[0].?);
    try testing.expectEqual(@as(usize, 1), cq.len());

    _ = try cq.enqueue("B");
    try testing.expectEqualStrings("A", cq.array[0].?);
    try testing.expectEqualStrings("B", cq.array[1].?);

    _ = try cq.enqueue("C");
    _ = try cq.enqueue("D");
    _ = try cq.enqueue("E");
    try testing.expectError(error.QueueFull, cq.enqueue("F"));

    try testing.expectEqualStrings("A", try cq.dequeue());
    try testing.expectEqual(@as(usize, 4), cq.len());
    try testing.expectEqualStrings("B", cq.first().?);

    try testing.expectEqualStrings("B", try cq.dequeue());
    try testing.expectEqualStrings("C", try cq.dequeue());
    try testing.expectEqualStrings("D", try cq.dequeue());
    try testing.expectEqualStrings("E", try cq.dequeue());
    try testing.expectError(error.Underflow, cq.dequeue());
}

test "circular queue: wrap-around behavior" {
    var cq = try CircularQueue(i32).init(testing.allocator, 4);
    defer cq.deinit();

    _ = try cq.enqueue(1);
    _ = try cq.enqueue(2);
    _ = try cq.enqueue(3);
    try testing.expectEqual(@as(i32, 1), try cq.dequeue());
    try testing.expectEqual(@as(i32, 2), try cq.dequeue());

    _ = try cq.enqueue(4);
    _ = try cq.enqueue(5);

    try testing.expectEqual(@as(i32, 3), try cq.dequeue());
    try testing.expectEqual(@as(i32, 4), try cq.dequeue());
    try testing.expectEqual(@as(i32, 5), try cq.dequeue());
    try testing.expect(cq.isEmpty());
}

test "circular queue: extreme long cycle" {
    var cq = try CircularQueue(i64).init(testing.allocator, 256);
    defer cq.deinit();

    var in: i64 = 0;
    var out: i64 = 0;

    while (in < 20_000) : (in += 1) {
        _ = try cq.enqueue(in);
        if (cq.len() > 128) {
            try testing.expectEqual(out, try cq.dequeue());
            out += 1;
        }
    }

    while (!cq.isEmpty()) {
        try testing.expectEqual(out, try cq.dequeue());
        out += 1;
    }

    try testing.expectEqual(in, out);
}
